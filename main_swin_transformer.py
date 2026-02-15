#import matplotlib.pyplot as plt
#import numpy as np
#import tensorflow as tf
#import tensorflow_addons as tfa
#from tensorflow import keras
#from tensorflow.keras import layers
#import os
#from pathlib import Path
#import cv2
#
## ============================================================================
## DATA LOADING FOR YOUR DATASET
## ============================================================================
#
#def parse_yolo_label(label_path, img_width, img_height):
#    """Parse YOLO format label file and extract class labels"""
#    classes = []
#    if os.path.exists(label_path):
#        with open(label_path, 'r') as f:
#            for line in f:
#                parts = line.strip().split()
#                if len(parts) >= 1:
#                    class_id = int(parts[0])
#                    classes.append(class_id)
#    # Return the first class (or 0 if no labels)
#    return classes[0] if classes else 0
#
#def load_apple_leaf_dataset(data_dir, img_size=(224, 224)):
#    """
#    Load apple leaf dataset from directory structure
#    data_dir should contain train/, valid/, and test/ folders
#    """
#    images = []
#    labels = []
#    
#    img_dir = os.path.join(data_dir, 'images')
#    label_dir = os.path.join(data_dir, 'labels')
#    
#    if not os.path.exists(img_dir):
#        print(f"Warning: {img_dir} not found")
#        return np.array([]), np.array([])
#    
#    # Get all image files
#    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
#    
#    for img_file in img_files:
#        # Load image
#        img_path = os.path.join(img_dir, img_file)
#        img = cv2.imread(img_path)
#        if img is None:
#            continue
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        img = cv2.resize(img, img_size)
#        
#        # Load corresponding label
#        label_file = os.path.splitext(img_file)[0] + '.txt'
#        label_path = os.path.join(label_dir, label_file)
#        label = parse_yolo_label(label_path, img.shape[1], img.shape[0])
#        
#        images.append(img)
#        labels.append(label)
#    
#    return np.array(images), np.array(labels)
#
## ============================================================================
## HYPERPARAMETERS - ADJUST THESE FOR YOUR DATASET
## ============================================================================
#
## Dataset configuration
#dataset_path = 'appleleaf'  # Path to your dataset
#num_classes = 9  # Change this to match your number of apple leaf disease classes
#input_shape = (224, 224, 3)  # Larger input size for better feature extraction
#
## Model hyperparameters
#patch_size = (4, 4)  # 4-by-4 sized patches
#dropout_rate = 0.1
#num_heads = 8
#embed_dim = 96  # Embedding dimension
#num_mlp = 384  # MLP layer size
#qkv_bias = True
#window_size = 7  # Size of attention window
#shift_size = 3  # Size of shifting window
#image_dimension = 224
#
#num_patch_x = input_shape[0] // patch_size[0]
#num_patch_y = input_shape[1] // patch_size[1]
#
## Training hyperparameters
#learning_rate = 1e-4
#batch_size = 16  # Smaller batch size for larger images
#num_epochs = 100
#weight_decay = 0.0001
#label_smoothing = 0.1
#
#print(f"Number of patches: {num_patch_x} x {num_patch_y} = {num_patch_x * num_patch_y}")
#
## ============================================================================
## LOAD YOUR DATASET
## ============================================================================
#
#print("Loading training data...")
#x_train, y_train = load_apple_leaf_dataset(
#    os.path.join(dataset_path, 'train'), 
#    img_size=(input_shape[0], input_shape[1])
#)
#
#print("Loading validation data...")
#x_valid, y_valid = load_apple_leaf_dataset(
#    os.path.join(dataset_path, 'valid'),
#    img_size=(input_shape[0], input_shape[1])
#)
#
#print("Loading test data...")
#x_test, y_test = load_apple_leaf_dataset(
#    os.path.join(dataset_path, 'test'),
#    img_size=(input_shape[0], input_shape[1])
#)
#
## Normalize images
#x_train, x_valid, x_test = x_train / 255.0, x_valid / 255.0, x_test / 255.0
#
## Convert labels to one-hot encoding
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_valid = keras.utils.to_categorical(y_valid, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#
#print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
#print(f"x_valid shape: {x_valid.shape} - y_valid shape: {y_valid.shape}")
#print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
#
## Visualize some samples
#plt.figure(figsize=(10, 10))
#for i in range(min(25, len(x_train))):
#    plt.subplot(5, 5, i + 1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(x_train[i])
#    plt.title(f"Class: {np.argmax(y_train[i])}")
#plt.tight_layout()
#plt.show()
#
## ============================================================================
## HELPER FUNCTIONS
## ============================================================================
#
#def window_partition(x, window_size):
#    _, height, width, channels = x.shape
#    patch_num_y = height // window_size
#    patch_num_x = width // window_size
#    x = tf.reshape(
#        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
#    )
#    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
#    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
#    return windows
#
#def window_reverse(windows, window_size, height, width, channels):
#    patch_num_y = height // window_size
#    patch_num_x = width // window_size
#    x = tf.reshape(
#        windows,
#        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
#    )
#    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
#    x = tf.reshape(x, shape=(-1, height, width, channels))
#    return x
#
#class DropPath(layers.Layer):
#    def __init__(self, drop_prob=None, **kwargs):
#        super(DropPath, self).__init__(**kwargs)
#        self.drop_prob = drop_prob
#
#    def call(self, x, training=None):
#        if self.drop_prob == 0.0 or not training:
#            return x
#        input_shape = tf.shape(x)
#        batch_size = input_shape[0]
#        rank = x.shape.rank
#        shape = (batch_size,) + (1,) * (rank - 1)
#        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
#        path_mask = tf.floor(random_tensor)
#        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
#        return output
#
## ============================================================================
## WINDOW ATTENTION
## ============================================================================
#
#class WindowAttention(layers.Layer):
#    def __init__(
#        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
#    ):
#        super(WindowAttention, self).__init__(**kwargs)
#        self.dim = dim
#        self.window_size = window_size
#        self.num_heads = num_heads
#        self.scale = (dim // num_heads) ** -0.5
#        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
#        self.dropout = layers.Dropout(dropout_rate)
#        self.proj = layers.Dense(dim)
#
#    def build(self, input_shape):
#        num_window_elements = (2 * self.window_size[0] - 1) * (
#            2 * self.window_size[1] - 1
#        )
#        self.relative_position_bias_table = self.add_weight(
#            shape=(num_window_elements, self.num_heads),
#            initializer=tf.initializers.Zeros(),
#            trainable=True,
#        )
#        coords_h = np.arange(self.window_size[0])
#        coords_w = np.arange(self.window_size[1])
#        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
#        coords = np.stack(coords_matrix)
#        coords_flatten = coords.reshape(2, -1)
#        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#        relative_coords = relative_coords.transpose([1, 2, 0])
#        relative_coords[:, :, 0] += self.window_size[0] - 1
#        relative_coords[:, :, 1] += self.window_size[1] - 1
#        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#        relative_position_index = relative_coords.sum(-1)
#
#        self.relative_position_index = tf.Variable(
#            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
#        )
#
#    def call(self, x, mask=None):
#        _, size, channels = x.shape
#        head_dim = channels // self.num_heads
#        x_qkv = self.qkv(x)
#        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
#        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
#        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
#        q = q * self.scale
#        k = tf.transpose(k, perm=(0, 1, 3, 2))
#        attn = q @ k
#
#        num_window_elements = self.window_size[0] * self.window_size[1]
#        relative_position_index_flat = tf.reshape(
#            self.relative_position_index, shape=(-1,)
#        )
#        relative_position_bias = tf.gather(
#            self.relative_position_bias_table, relative_position_index_flat
#        )
#        relative_position_bias = tf.reshape(
#            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
#        )
#        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
#        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
#
#        if mask is not None:
#            nW = mask.get_shape()[0]
#            mask_float = tf.cast(
#                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
#            )
#            attn = (
#                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
#                + mask_float
#            )
#            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
#            attn = keras.activations.softmax(attn, axis=-1)
#        else:
#            attn = keras.activations.softmax(attn, axis=-1)
#        attn = self.dropout(attn)
#
#        x_qkv = attn @ v
#        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
#        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
#        x_qkv = self.proj(x_qkv)
#        x_qkv = self.dropout(x_qkv)
#        return x_qkv
#
## ============================================================================
## SWIN TRANSFORMER BLOCK
## ============================================================================
#
#class SwinTransformer(layers.Layer):
#    def __init__(
#        self,
#        dim,
#        num_patch,
#        num_heads,
#        window_size=7,
#        shift_size=0,
#        num_mlp=1024,
#        qkv_bias=True,
#        dropout_rate=0.0,
#        **kwargs,
#    ):
#        super(SwinTransformer, self).__init__(**kwargs)
#
#        self.dim = dim
#        self.num_patch = num_patch
#        self.num_heads = num_heads
#        self.window_size = window_size
#        self.shift_size = shift_size
#        self.num_mlp = num_mlp
#
#        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
#        self.attn = WindowAttention(
#            dim,
#            window_size=(self.window_size, self.window_size),
#            num_heads=num_heads,
#            qkv_bias=qkv_bias,
#            dropout_rate=dropout_rate,
#        )
#        self.drop_path = DropPath(dropout_rate)
#        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
#
#        self.mlp = keras.Sequential(
#            [
#                layers.Dense(num_mlp),
#                layers.Activation(keras.activations.gelu),
#                layers.Dropout(dropout_rate),
#                layers.Dense(dim),
#                layers.Dropout(dropout_rate),
#            ]
#        )
#
#        if min(self.num_patch) < self.window_size:
#            self.shift_size = 0
#            self.window_size = min(self.num_patch)
#
#    def build(self, input_shape):
#        if self.shift_size == 0:
#            self.attn_mask = None
#        else:
#            height, width = self.num_patch
#            h_slices = (
#                slice(0, -self.window_size),
#                slice(-self.window_size, -self.shift_size),
#                slice(-self.shift_size, None),
#            )
#            w_slices = (
#                slice(0, -self.window_size),
#                slice(-self.window_size, -self.shift_size),
#                slice(-self.shift_size, None),
#            )
#            mask_array = np.zeros((1, height, width, 1))
#            count = 0
#            for h in h_slices:
#                for w in w_slices:
#                    mask_array[:, h, w, :] = count
#                    count += 1
#            mask_array = tf.convert_to_tensor(mask_array)
#
#            mask_windows = window_partition(mask_array, self.window_size)
#            mask_windows = tf.reshape(
#                mask_windows, shape=[-1, self.window_size * self.window_size]
#            )
#            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
#                mask_windows, axis=2
#            )
#            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
#            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
#            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)
#
#    def call(self, x, training=None):
#        height, width = self.num_patch
#        _, num_patches_before, channels = x.shape
#        x_skip = x
#        x = self.norm1(x)
#        x = tf.reshape(x, shape=(-1, height, width, channels))
#        if self.shift_size > 0:
#            shifted_x = tf.roll(
#                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
#            )
#        else:
#            shifted_x = x
#
#        x_windows = window_partition(shifted_x, self.window_size)
#        x_windows = tf.reshape(
#            x_windows, shape=(-1, self.window_size * self.window_size, channels)
#        )
#        attn_windows = self.attn(x_windows, mask=self.attn_mask)
#
#        attn_windows = tf.reshape(
#            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
#        )
#        shifted_x = window_reverse(
#            attn_windows, self.window_size, height, width, channels
#        )
#        if self.shift_size > 0:
#            x = tf.roll(
#                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
#            )
#        else:
#            x = shifted_x
#
#        x = tf.reshape(x, shape=(-1, height * width, channels))
#        x = self.drop_path(x, training=training)
#        x = x_skip + x
#        x_skip = x
#        x = self.norm2(x)
#        x = self.mlp(x)
#        x = self.drop_path(x, training=training)
#        x = x_skip + x
#        return x
#
## ============================================================================
## PATCH LAYERS
## ============================================================================
#
#class PatchExtract(layers.Layer):
#    def __init__(self, patch_size, **kwargs):
#        super(PatchExtract, self).__init__(**kwargs)
#        self.patch_size_x = patch_size[0]
#        self.patch_size_y = patch_size[1]
#
#    def call(self, images):
#        batch_size = tf.shape(images)[0]
#        patches = tf.image.extract_patches(
#            images=images,
#            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
#            strides=(1, self.patch_size_x, self.patch_size_y, 1),
#            rates=(1, 1, 1, 1),
#            padding="VALID",
#        )
#        patch_dim = patches.shape[-1]
#        patch_num = patches.shape[1]
#        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))
#
#class PatchEmbedding(layers.Layer):
#    def __init__(self, num_patch, embed_dim, **kwargs):
#        super(PatchEmbedding, self).__init__(**kwargs)
#        self.num_patch = num_patch
#        self.proj = layers.Dense(embed_dim)
#        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)
#
#    def call(self, patch):
#        pos = tf.range(start=0, limit=self.num_patch, delta=1)
#        return self.proj(patch) + self.pos_embed(pos)
#
#class PatchMerging(tf.keras.layers.Layer):
#    def __init__(self, num_patch, embed_dim):
#        super(PatchMerging, self).__init__()
#        self.num_patch = num_patch
#        self.embed_dim = embed_dim
#        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)
#
#    def call(self, x):
#        height, width = self.num_patch
#        _, _, C = x.get_shape().as_list()
#        x = tf.reshape(x, shape=(-1, height, width, C))
#        x0 = x[:, 0::2, 0::2, :]
#        x1 = x[:, 1::2, 0::2, :]
#        x2 = x[:, 0::2, 1::2, :]
#        x3 = x[:, 1::2, 1::2, :]
#        x = tf.concat((x0, x1, x2, x3), axis=-1)
#        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
#        return self.linear_trans(x)
#
## ============================================================================
## BUILD MODEL
## ============================================================================
#
#input_layer = layers.Input(input_shape)
#x = layers.RandomCrop(image_dimension, image_dimension)(input_layer)
#x = layers.RandomFlip("horizontal")(x)
#x = layers.RandomRotation(0.1)(x)  # Additional augmentation
#x = PatchExtract(patch_size)(x)
#x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
#
## First Swin Transformer block
#x = SwinTransformer(
#    dim=embed_dim,
#    num_patch=(num_patch_x, num_patch_y),
#    num_heads=num_heads,
#    window_size=window_size,
#    shift_size=0,
#    num_mlp=num_mlp,
#    qkv_bias=qkv_bias,
#    dropout_rate=dropout_rate,
#)(x)
#
## Second Swin Transformer block with shifted window
#x = SwinTransformer(
#    dim=embed_dim,
#    num_patch=(num_patch_x, num_patch_y),
#    num_heads=num_heads,
#    window_size=window_size,
#    shift_size=shift_size,
#    num_mlp=num_mlp,
#    qkv_bias=qkv_bias,
#    dropout_rate=dropout_rate,
#)(x)
#
#x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
#x = layers.GlobalAveragePooling1D()(x)
#x = layers.Dropout(0.3)(x)
#output = layers.Dense(num_classes, activation="softmax")(x)
#
#model = keras.Model(input_layer, output)
#model.summary()
#
## ============================================================================
## COMPILE AND TRAIN MODEL
## ============================================================================
#
#model.compile(
#    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
#    optimizer=tfa.optimizers.AdamW(
#        learning_rate=learning_rate, weight_decay=weight_decay
#    ),
#    metrics=[
#        keras.metrics.CategoricalAccuracy(name="accuracy"),
#        keras.metrics.TopKCategoricalAccuracy(3, name="top-3-accuracy"),
#    ],
#)
#
## Callbacks
#import datetime
#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#
#callbacks = [
#    keras.callbacks.EarlyStopping(
#        monitor='val_loss', 
#        patience=15, 
#        restore_best_weights=True
#    ),
#    keras.callbacks.ReduceLROnPlateau(
#        monitor='val_loss',
#        factor=0.5,
#        patience=5,
#        min_lr=1e-7
#    ),
#    keras.callbacks.ModelCheckpoint(
#        f'best_swin_model_{timestamp}.keras',  # Use .keras format and timestamp
#        monitor='val_accuracy',
#        save_best_only=True,
#        save_weights_only=False
#    )
#]
#
## Train the model
#history = model.fit(
#    x_train,
#    y_train,
#    batch_size=batch_size,
#    epochs=num_epochs,
#    validation_data=(x_valid, y_valid),
#    callbacks=callbacks,
#)
#
## ============================================================================
## VISUALIZE TRAINING
## ============================================================================
#
#plt.figure(figsize=(12, 4))
#
#plt.subplot(1, 2, 1)
#plt.plot(history.history["loss"], label="train_loss")
#plt.plot(history.history["val_loss"], label="val_loss")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Train and Validation Losses Over Epochs")
#plt.legend()
#plt.grid()
#
#plt.subplot(1, 2, 2)
#plt.plot(history.history["accuracy"], label="train_accuracy")
#plt.plot(history.history["val_accuracy"], label="val_accuracy")
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
#plt.title("Train and Validation Accuracy Over Epochs")
#plt.legend()
#plt.grid()
#
#plt.tight_layout()
#plt.show()
#
## ============================================================================
## EVALUATE ON TEST SET
## ============================================================================
#
#loss, accuracy, top_3_accuracy = model.evaluate(x_test, y_test)
#print(f"\nTest Results:")
#print(f"Test loss: {round(loss, 4)}")
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#print(f"Test top-3 accuracy: {round(top_3_accuracy * 100, 2)}%")
#
## Save the final model
#model.save(f'final_swin_transformer_apple_leaf_{timestamp}.keras')
#print(f"\nModel saved as 'final_swin_transformer_apple_leaf_{timestamp}.keras'")


import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import timm  # For Swin Transformer

class BalancedAppleLeafDataset(Dataset):
    """Dataset class with support for class balancing"""
    
    def __init__(self, data_dir, split='train', transform=None, balance_method='oversample'):
        """
        Args:
            data_dir: Base directory containing the dataset
            split: 'train', 'val', or 'test'
            transform: Torchvision transforms
            balance_method: 'oversample', 'undersample', or 'none'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.balance_method = balance_method
        
        # Load YAML config
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.class_names = config['names']
        self.n_classes = config['nc']
        
        # Get paths
        if split == 'val':
            split = 'valid' if 'valid' in config else 'val'
        
        img_path = config[split].replace('../', '')
        self.img_dir = self.data_dir / img_path
        self.label_dir = Path(str(self.img_dir).replace('images', 'labels').replace('image', 'labels'))
        
        # Load all image paths and labels
        self.samples = []
        self.labels = []
        self._load_dataset()
        
        # Balance the dataset if training
        if split == 'train' and balance_method != 'none':
            self._balance_dataset()
        
        print(f"\n{split.upper()} Dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Class distribution:")
        class_counts = Counter(self.labels)
        for class_id in range(self.n_classes):
            count = class_counts.get(class_id, 0)
            print(f"    Class {self.class_names[class_id]}: {count}")
    
    def _load_dataset(self):
        """Load all image paths and labels"""
        image_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))
        
        for img_path in image_files:
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        self.samples.append(str(img_path))
                        self.labels.append(class_id)
    
    def _balance_dataset(self):
        """Balance the dataset using oversampling or undersampling"""
        class_counts = Counter(self.labels)
        
        if self.balance_method == 'oversample':
            # Oversample minority classes to match majority class
            max_count = max(class_counts.values())
            
            print(f"\n{'='*60}")
            print(f"BALANCING DATASET (Oversampling)")
            print(f"{'='*60}")
            print(f"Target samples per class: {max_count}")
            
            balanced_samples = []
            balanced_labels = []
            
            for class_id in range(self.n_classes):
                # Get all samples for this class
                class_indices = [i for i, label in enumerate(self.labels) if label == class_id]
                class_samples = [self.samples[i] for i in class_indices]
                
                current_count = len(class_samples)
                
                # Oversample by repeating samples
                if current_count > 0:
                    repeats = max_count // current_count
                    remainder = max_count % current_count
                    
                    # Add repeated samples
                    balanced_samples.extend(class_samples * repeats)
                    balanced_labels.extend([class_id] * (current_count * repeats))
                    
                    # Add random samples for remainder
                    if remainder > 0:
                        import random
                        random_samples = random.choices(class_samples, k=remainder)
                        balanced_samples.extend(random_samples)
                        balanced_labels.extend([class_id] * remainder)
                    
                    print(f"  Class {self.class_names[class_id]}: {current_count} ? {max_count} (+{max_count - current_count})")
            
            self.samples = balanced_samples
            self.labels = balanced_labels
            
        elif self.balance_method == 'undersample':
            # Undersample majority classes to match minority class
            min_count = min(class_counts.values())
            
            print(f"\n{'='*60}")
            print(f"BALANCING DATASET (Undersampling)")
            print(f"{'='*60}")
            print(f"Target samples per class: {min_count}")
            
            balanced_samples = []
            balanced_labels = []
            
            for class_id in range(self.n_classes):
                # Get all samples for this class
                class_indices = [i for i, label in enumerate(self.labels) if label == class_id]
                class_samples = [self.samples[i] for i in class_indices]
                
                current_count = len(class_samples)
                
                # Undersample by random selection
                if current_count > 0:
                    import random
                    selected_samples = random.sample(class_samples, min(min_count, current_count))
                    balanced_samples.extend(selected_samples)
                    balanced_labels.extend([class_id] * len(selected_samples))
                    
                    print(f"  Class {self.class_names[class_id]}: {current_count} ? {len(selected_samples)} (-{current_count - len(selected_samples)})")
            
            self.samples = balanced_samples
            self.labels = balanced_labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SwinTransformerClassifier:
    """Swin Transformer for Apple Leaf Classification with Data Balancing"""
    
    def __init__(self, data_dir, n_classes=9, img_size=224, balance_method='oversample'):
        """
        Args:
            data_dir: Base directory containing the dataset
            n_classes: Number of classes
            img_size: Input image size
            balance_method: 'oversample', 'undersample', or 'weighted_loss'
        """
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.img_size = img_size
        self.balance_method = balance_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"SWIN TRANSFORMER INITIALIZATION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Number of classes: {n_classes}")
        print(f"Balance method: {balance_method}")
        print(f"{'='*60}\n")
        
        # Load YAML config
        yaml_path = Path(data_dir) / 'data.yaml'
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        self.class_names = config['names']
        
        # Initialize model
        self.model = self._create_model()
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        if balance_method == 'weighted_loss':
            # Use weighted loss instead of data balancing
            self.train_dataset = BalancedAppleLeafDataset(
                data_dir, 'train', self.train_transform, balance_method='none'
            )
            self.class_weights = self._calculate_class_weights()
        else:
            # Use oversampling or undersampling
            self.train_dataset = BalancedAppleLeafDataset(
                data_dir, 'train', self.train_transform, balance_method=balance_method
            )
            self.class_weights = None
        
        self.val_dataset = BalancedAppleLeafDataset(
            data_dir, 'val', self.val_transform, balance_method='none'
        )
        self.test_dataset = BalancedAppleLeafDataset(
            data_dir, 'test', self.val_transform, balance_method='none'
        )
    
    def _create_model(self):
        """Create Swin Transformer model"""
        # Load pre-trained Swin Transformer
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=self.n_classes)
        
        # You can also use other variants:
        # model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=self.n_classes)
        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=self.n_classes)
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: Swin Transformer Tiny")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        return model
    
    def _calculate_class_weights(self):
        """Calculate class weights for weighted loss"""
        class_counts = Counter(self.train_dataset.labels)
        total_samples = len(self.train_dataset.labels)
        
        weights = []
        for class_id in range(self.n_classes):
            count = class_counts.get(class_id, 1)
            weight = total_samples / (self.n_classes * count)
            weights.append(weight)
        
        weights = torch.FloatTensor(weights).to(self.device)
        
        print(f"\nClass weights for weighted loss:")
        for i, w in enumerate(weights):
            print(f"  Class {self.class_names[i]}: {w:.4f}")
        
        return weights
    
    def train(self, epochs=50, batch_size=32, lr=1e-4, save_path='swin_best_model.pth'):
        """Train the model"""
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Loss function
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"\n{'='*60}")
        print(f"TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, labels in train_pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for images, labels in val_pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }, save_path)
                print(f"  ? Best model saved! (Val Acc: {val_acc:.2f}%)")
            
            print(f"{'='*60}\n")
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, split='test'):
        """Evaluate the model"""
        if split == 'test':
            dataset = self.test_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.train_dataset
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f'Evaluating {split}'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"\n{'='*60}")
        print(f"{split.upper()} SET EVALUATION")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {split.upper()} Set')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{split}.png', dpi=300)
        plt.show()
        
        return acc, all_preds, all_labels
    
    def plot_training_history(self, history):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves_swin.png', dpi=300)
        plt.show()
    
    def load_model(self, path='swin_best_model.pth'):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    data_dir = "/cabinfs/home/user1/Rahul_project/project_apple_roboflow/appleleaf"
    
    # Choose balancing method:
    # 'oversample' - Duplicate minority class samples to match majority
    # 'undersample' - Randomly reduce majority class samples to match minority
    # 'weighted_loss' - Use weighted CrossEntropyLoss (no data modification)
    balance_method = 'oversample'  # Change this as needed
    
    # Initialize classifier
    classifier = SwinTransformerClassifier(
        data_dir=data_dir,
        n_classes=9,
        img_size=224,
        balance_method=balance_method
    )
    
    # Train the model
    history = classifier.train(
        epochs=50,
        batch_size=32,
        lr=1e-4,
        save_path='swin_balanced_best.pth'
    )
    
    # Plot training curves
    classifier.plot_training_history(history)
    
    # Evaluate on validation set
    classifier.evaluate(split='val')
    
    # Evaluate on test set
    classifier.evaluate(split='test')
    
    print("\n? Training and evaluation complete!")