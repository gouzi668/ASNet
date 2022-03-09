#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)
print('tf version ',tf.__version__)   
#%%
class BasicBlock(keras.Model):
    def __init__(self,filters,elips,**kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.elips = elips
        self.filters = filters
        self.conv1 = layers.Conv2D(filters=self.filters,kernel_size=3,strides=1,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=self.filters,kernel_size=3,strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.maxpool = layers.MaxPool2D(padding='same')
        self.averPool = layers.AveragePooling2D(padding='same')
        self.block_name = kwargs['name']
        self.mask = layers.Conv2D(filters=7,kernel_size=1,strides=1,padding='same',activation='softmax',name=self.block_name +'mask_conv')
        self.mask2 = layers.Conv2D(filters=7,kernel_size=1,strides=1,padding='same',activation='softmax',name=self.block_name +'mask_conv2')
    def __call__(self,inputs,training=None,masked=False):
        val = self.conv1(inputs)
        val = self.bn1(val)
        val = self.relu1(val)

        if masked:
            mask = self.mask(val)
            mask = mask * tf.constant([-0.8,1.,1.,1.,1.,1.,1.],dtype=tf.float32)
            mask = tf.reduce_sum(mask,axis=-1,keepdims=True) 
            mask = layers.ReLU()(mask)
            val = val * mask
 
        val = self.conv2(val)
        val = self.bn2(val)
        val = self.relu2(val)
        if masked:
            #prt = self.projection(val)
            mask = self.mask2(val)
            mask = mask * tf.constant([-0.8,1.,1.,1.,1.,1.,1.],dtype=tf.float32)
            mask = tf.reduce_sum(mask,axis=-1,keepdims=True) 
            mask = layers.ReLU()(mask)
            val = val * mask 
            out = self.maxpool(val) 
            #out = out * mask
        else:      
            out = self.maxpool(val)
        return out

def make_model(num_class,name='apfNet'):
    inputs = keras.Input(shape=(128,128,3),dtype=tf.float32)
    out1 = BasicBlock(16,1e-18,name='block1')(inputs,masked=False)
    out2 = BasicBlock(24,1e-18,name='block2')(out1,masked=True)
    out3 = BasicBlock(24,1e-18,name='block3')(out2,masked=False)
    out4 = layers.Conv2D(32,3,padding='same')(out3)
    out4 = layers.BatchNormalization()(out4)
    out4 = layers.ReLU()(out4)
    out4 = layers.Conv2D(32,3,padding='same')(out4)
    out4 = layers.BatchNormalization()(out4)
    out4 = layers.ReLU()(out4)
    feature_out = layers.MaxPooling2D(padding='same')(out4)
    out = layers.GlobalAveragePooling2D()(feature_out)
    out = layers.Dense(16, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(16, activation='relu')(out)
    out = layers.Dense(num_class)(out)
    return keras.Model(inputs,out,name=name)
model = make_model(6)
# model.summary()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#%%
data_augumantation = keras.Sequential([
  layers.RandomFlip(mode="horizontal_and_vertical"),
  layers.RandomContrast(factor=0.2),
  layers.RandomRotation(factor=(-0.02, 0.02)),
  layers.Rescaling(scale=1./255,offset=-1.)
])
# data_augumantation = layers.RandomFlip(mode="horizontal_and_vertical")

class GetTrainModel(keras.Model):
    def __init__(self,augu,model,**kwargs):
        super(GetTrainModel,self).__init__(**kwargs)
        self.augu = augu
        self.model = model
    def train_step(self,data):
        x,y = data
        x = self.augu(x)
        with tf.GradientTape() as tape:
            y_pred = self(x,training = True)
            loss = self.compiled_loss(y,y_pred,regularization_losses=self.losses)
            train_vals = self.trainable_weights
            gradients = tape.gradient(loss,train_vals)
        self.optimizer.apply_gradients(zip(gradients,train_vals))
        self.compiled_metrics.update_state(y,y_pred)
        return {m.name:m.result() for m in self.metrics}
    def test_step(self,data):
      x,y = data
      y_pred = self(x,training = False)
      self.compiled_loss(y,y_pred,regularization_losses=self.losses)
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}
    def summary(self):
        self.model.summary()
    def save_weights(self,filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(filepath,overwrite,save_format,options)
    def load_weights(self,filepath, by_name=False, skip_mismatch=False, options=None):
        self.model.load_weights(filepath,by_name,skip_mismatch,options)
    def get_config(self):
        config = super(GetTrainModel,self).get_config()
        config.update({"augu":self.augu,"model":self.model})
        return config
    def call(self,inputs,training = None):
        return self.model(inputs)
def scheduler(epoch, lr):
    times = [100,200,300,400]
    if epoch in times:
        return lr / 1.778279
    else:
        return lr
#%%
num_trains = 40
seeds = [10086375,  5172380, 16411008, 26500616, 18097334, 29958170,  4239313,  2496542,
 20007066, 18623981, 11329605,  5293286,  6619790,  1922278, 20292963, 16545387,
 13375195, 24750856, 23206384,   535534, 18030313,  4101250, 28685047,  6133629,
 13109026, 22635734, 17600701, 14738162, 25554058, 23510864, 19751270, 29349700,
  6982881, 14593532,  2635328, 17060469, 16364842,  3348582, 26094771,  7502465]
for i in range(num_trains):
    print('.................................in {0} training processing........................'.format(i+1))
    #seed = np.random.randint(low=300,high=30000000,size=1)[0]
    seed = seeds[i]
    data_dir = r''
    batch_size = 64
    img_size = (128,128)
    train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed = seed,    
    image_size=img_size,
    interpolation = 'bicubic',
    batch_size=batch_size)


    val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",  
    seed= seed,
    image_size=img_size,
    interpolation = 'bicubic',
    batch_size=batch_size)

    example_data = next(iter(train_ds))
    normalization_layer = layers.Rescaling(1./255,offset=-1.)
    train_ds = train_ds.map(lambda x,y : (tf.cast(x,dtype=tf.float32),y))
    #train_ds = train_ds.map(lambda x,y : (normalization_layer(x),y))
    val_ds = val_ds.map(lambda x,y : (normalization_layer(x),y))
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) 
    example_data = next(iter(val_ds)) 
    initial_learning_rate = 1e-3
    decay_steps = 350
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    model = make_model(6)
    trainmodel = GetTrainModel(data_augumantation,model)
    trainmodel.summary()
    checkpoint_filepath = r''
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    callback = tf.keras.callbacks.LearningRateScheduler(lr_decayed_fn)
    SaveWightsCallback = keras.callbacks.ModelCheckpoint(filepath=r'weights',monitor='val_loss',verbose=1,save_best_only=True)
    trainmodel.compile(
    optimizer=optimizers.Adam(learning_rate=initial_learning_rate),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    history = trainmodel.fit(train_ds,validation_data = val_ds,epochs=300,validation_freq =1,callbacks=[callback,model_checkpoint_callback])
    trainmodel.load_weights(checkpoint_filepath)
    acc = trainmodel.evaluate(val_ds)[1]
    with open(r'','a+') as f:
        f.write(str(acc) + " ")

