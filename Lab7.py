import os
import zipfile
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, utils, losses

ARCHIVE_FILE = "cats.zip"
WORK_DIR = "cats_extracted"
RESULT_DIR = "training_output"
IMG_W, IMG_H = 64, 64
BATCH_SIZE = 64
Z_DIM = 128
MAX_EPOCHS = 30
CHECKPOINT_FREQ = 1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def prepare_environment():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    
    if not any(os.scandir(WORK_DIR)):
        if not os.path.exists(ARCHIVE_FILE):
            raise FileNotFoundError(f"Missing file: {ARCHIVE_FILE}")
        
        with zipfile.ZipFile(ARCHIVE_FILE, "r") as zf:
            zf.extractall(WORK_DIR)

    os.makedirs(RESULT_DIR, exist_ok=True)

def locate_images(base_path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    target_dir = None
    max_files = 0

    for current_root, _, filenames in os.walk(base_path):
        count = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in valid_exts)
        if count > max_files:
            max_files = count
            target_dir = current_root
            
    if not target_dir:
        raise RuntimeError("No valid images found in the extracted directory.")
        
    return target_dir

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_W, IMG_H])
    img = (img - 127.5) / 127.5
    return img

def create_pipeline(data_root):
    dataset = utils.image_dataset_from_directory(
        data_root,
        labels=None,
        label_mode=None,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_W, IMG_H),
        shuffle=True,
        smart_resize=True
    )
    
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def define_generator():
    model = models.Sequential()
    model.add(layers.Input(shape=(Z_DIM,)))
    model.add(layers.Dense(4 * 4 * 512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((4, 4, 512)))

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding="same", activation="tanh"))
    return model

def define_discriminator():
    model = models.Sequential()
    model.add(layers.Input(shape=(IMG_W, IMG_H, 3)))
    
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def calculate_disc_loss(real_out, fake_out):
    loss_real = cross_entropy(tf.ones_like(real_out), real_out)
    loss_fake = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return loss_real + loss_fake

def calculate_gen_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)

def export_samples(model, input_seeds, epoch_idx):
    predictions = model(input_seeds, training=False)
    predictions = (predictions * 0.5) + 0.5
    predictions = tf.clip_by_value(predictions, 0.0, 1.0)

    grid_size = int(np.sqrt(predictions.shape[0]))
    final_image_rows = []
    
    for row in range(grid_size):
        start_idx = row * grid_size
        end_idx = start_idx + grid_size
        row_images = tf.concat([predictions[i] for i in range(start_idx, end_idx)], axis=1)
        final_image_rows.append(row_images)
        
    full_grid = tf.concat(final_image_rows, axis=0)
    save_path = os.path.join(RESULT_DIR, f"sample_epoch_{epoch_idx}.png")
    utils.save_img(save_path, full_grid)

@tf.function
def perform_training_step(real_imgs, generator, discriminator, opt_g, opt_d):
    batch_dim = tf.shape(real_imgs)[0]
    random_vector = tf.random.normal([batch_dim, Z_DIM])

    with tf.GradientTape() as tape_d, tf.GradientTape() as tape_g:
        generated_imgs = generator(random_vector, training=True)

        logits_real = discriminator(real_imgs, training=True)
        logits_fake = discriminator(generated_imgs, training=True)

        loss_d = calculate_disc_loss(logits_real, logits_fake)
        loss_g = calculate_gen_loss(logits_fake)

    grads_d = tape_d.gradient(loss_d, discriminator.trainable_variables)
    grads_g = tape_g.gradient(loss_g, generator.trainable_variables)

    opt_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))

    return loss_d, loss_g

def run_training():
    prepare_environment()
    image_folder = locate_images(WORK_DIR)
    dataset = create_pipeline(image_folder)

    gen_model = define_generator()
    disc_model = define_discriminator()

    optimizer_g = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    optimizer_d = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    test_seed = tf.random.normal([16, Z_DIM])

    print("Starting training process...")

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()
        d_loss_history = []
        g_loss_history = []

        for batch_imgs in dataset:
            d_l, g_l = perform_training_step(batch_imgs, gen_model, disc_model, optimizer_g, optimizer_d)
            d_loss_history.append(d_l)
            g_loss_history.append(g_l)

        avg_d = np.mean(d_loss_history)
        avg_g = np.mean(g_loss_history)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{MAX_EPOCHS} -> D_Loss: {avg_d:.4f}, G_Loss: {avg_g:.4f} [{elapsed:.1f}s]")

        if epoch % CHECKPOINT_FREQ == 0:
            export_samples(gen_model, test_seed, epoch)

    gen_model.save(os.path.join(RESULT_DIR, "final_generator.keras"))
    disc_model.save(os.path.join(RESULT_DIR, "final_discriminator.keras"))

if __name__ == "__main__":
    run_training()