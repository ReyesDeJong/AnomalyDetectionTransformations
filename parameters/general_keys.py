"""Module that defines miscellaneous keys to manipulate the model"""

# For fitting (keep best models in validation)
ITERATION = 'iteration'
LOSS = 'loss'
DISC_LOSS = 'disc_loss'

# Names of metrics
ACC = 'acc'
ACCURACY = 'accuracy'

# Names of scores
DIRICHLET = 'dirichlet'
MATRIX_TRACE = 'matrix_trace'
ENTROPY = 'entropy'
CROSS_ENTROPY = 'cross_entropy'


# To evaluate the metrics
BATCHES_MEAN = 'batch_mean'
VALUES_PER_BATCH = 'values_per_batch'
MEAN = 'mean'
STD = 'std'

#set names
TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

#Types of data
REAL = 'real'
GENERATED = 'generated'
FAKE_IMAGE = 'fake_image'
REAL_IMAGE = 'real_image'

#summary keys
MERGED_IMAGE_SUMM = 'merged_image_summ'

#types of variable in data
CLASS = 'class'
MAGNITUDE = 'magnitude'
TIME = 'time'
ORIGINAL_MAGNITUDE_RANDOM = 'original_magnitude_random'
TIME_RANDOM = 'time_random'
GENERATED_MAGNITUDE = 'generated_magnitude'
IMAGES = 'images'
LABELS = 'labels'

#noise vector names
IN_NORMAL_RADIUS = 'in_normal_radius'
OUT_NORMAL_RADIUS = 'out_normal_radius'

# random things
RANDOM_SEED = 'random_seed'

# optimizer names
ADAM = 'adam'

# losses names
CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
