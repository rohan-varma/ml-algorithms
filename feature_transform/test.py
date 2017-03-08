from feature_transformer import FeatureTransformer
import numpy as np
ft = FeatureTransformer(degree=2, decay = 0.5)
X = np.arange(12)
X = ft.apply_decay(X)
print X
X = ft.generate_features(X)
