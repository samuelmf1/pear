import itertools
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Embedding, Flatten, Reshape
import keras.backend as K

def get_ctm(atac_pivot_df):
  ATACABLE_GENE_NAMES = set(atac_pivot_df.gene).intersection(GENE_NAMES)
  ADD_THESE_IN = set(GENE_NAMES) - set(atac_pivot_df.gene)
  print(len(ATACABLE_GENE_NAMES), len(ADD_THESE_IN))

  # List of columns to sum
  cell_types = ["B cells", "Myeloid cells", "NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]

  result_df = atac_pivot_df.groupby("gene")[cell_types].sum().reset_index().round(3)
  print(result_df.shape)
  result_df = result_df[result_df['gene'].isin(ATACABLE_GENE_NAMES)].set_index('gene')
  print(result_df.shape)

  # Now I need to add in all GENES that are also in the de_df file
  column_names = list(result_df.columns)
  row_indices = list(ADD_THESE_IN)
  missings_df = pd.DataFrame(0, index=row_indices, columns=column_names)
  for_embedding_df = pd.concat([result_df, missings_df]).reset_index()

  # Impute those missing values with cell-type avgs
  sorted_for_embedding_df = for_embedding_df.set_index('index').loc[GENE_NAMES].reset_index()
  column_avg = sorted_for_embedding_df.mean()
  sorted_for_embedding_df.replace(0, column_avg, inplace=True)

  cell_type_mappings = {}
  for ct in sorted_for_embedding_df.columns:
    if ct=='index':
      continue
    cell_type_mappings[ct] = np.array(sorted_for_embedding_df.loc[:, ct])

  return cell_type_mappings

def mean_rowwise_rmse(y_true, y_pred):
    squared_error = K.square(y_true - y_pred)
    rowwise_squared_error = K.mean(squared_error, axis=1)
    rowwise_rmse = K.sqrt(rowwise_squared_error)

    # calculate mean of rowwise root mean squared error
    mean_rowwise_rmse = K.mean(rowwise_rmse)

    return mean_rowwise_rmse

def build_model_atac(n_celltype_factors, n_sm_factors, n_nodes, n_genes, n_layers, ctm):

    celltype_input = Input(shape=(n_celltypes,), name="celltype_input")
    celltype_embedding = Embedding(n_celltypes, n_celltype_factors,
      weights=[ctm], input_length=1, trainable=True, name="celltype_embedding")
    celltype = Flatten()(celltype_embedding(celltype_input))

    sm_input = Input(shape=(n_sm,), name="sm_input")
    sm_embedding = Dense(n_sm_factors, name="sm_embedding")(sm_input)
    sm = Dense(n_sm_factors, name="sm")(sm_embedding)

    concatenated = concatenate([celltype, sm])
    for i in range(n_layers):
        concatenated = Dense(n_nodes, activation='relu', name="dense_{}".format(i))(concatenated)

    output_layer = Dense(n_genes, name="output")(concatenated)

    model = Model(inputs=[celltype_input, sm_input], outputs=output_layer)
    return model

# ==========
gene_expr_imputed_df = pd.read_parquet('/content/drive/My Drive/ML4FG_final/de_train.parquet')
GENE_NAMES = gene_expr_imputed_df.columns[5:]
cols = list(gene_expr_imputed_df.columns)
gene_expr_imputed_df.drop(columns=cols[2:5], inplace=True)

atac_pivot_df = pd.read_table('/content/drive/My Drive/ML4FG_final/atac_pivot.tsv')
ctm = get_ctm(atac_pivot_df)

cell_types = list(set(gene_expr_imputed_df.cell_type))
small_molecules = list(set(gene_expr_imputed_df.sm_name))
name_representations = cell_types + small_molecules

n_celltypes = gene_expr_imputed_df.cell_type.value_counts().shape[0] # Number of unique cell types = 6
n_sm = gene_expr_imputed_df.sm_name.value_counts().shape[0] # Number of unique sm = 144
n_genes = gene_expr_imputed_df.shape[1] - 2  # Number of genes
n_celltype_factors = len(ctm.keys()) # 6
n_sm_factors = 128
n_nodes = 256

# ==========
# combine arrays into a single matrix
data_matrix = np.vstack(list(ctm.values()))

pca = PCA(n_components=6)
reduced_data = pca.fit_transform(data_matrix)

for i, key in enumerate(ctm.keys()):
    ctm[key] = reduced_data[:, i]

data_matrix_ctm = np.vstack(list(ctm.values()))

# ==========
# one hot encode
encoded_data = pd.get_dummies(gene_expr_imputed_df, columns=['cell_type', 'sm_name'])

x_cols = encoded_data.columns[-(n_celltypes+n_sm):]
y_cols = filter(lambda x : x not in x_cols, encoded_data.columns)

X = encoded_data.loc[:, x_cols]
y = encoded_data.loc[:, y_cols]

# normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_layer_val = None
best_mrrmse = float('inf')

for n_layers in [2]:
  accuracy_values = []
  mrrmse_values = []
  for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = build_model_atac(n_celltype_factors, n_sm_factors, n_nodes, n_genes, n_layers, data_matrix_ctm)
    model.compile(optimizer='adam', loss=mean_rowwise_rmse, metrics=['accuracy'])

    # train model
    trained_model = model.fit([X_train_fold[:, :n_celltypes], X_train_fold[:, n_celltypes:]], y_train_fold, epochs=1000, batch_size=32, verbose=0)
    accuracy_values.append(np.mean(trained_model.history.get('accuracy', 0.5)))

    # evaluate model on validation set
    mrrmse = model.evaluate([X_val_fold[:, :n_celltypes], X_val_fold[:, n_celltypes:]], y_val_fold, verbose=0)
    mrrmse_values.append(mrrmse)

  average_mrrmse = np.mean(mrrmse_values)
  average_acc = np.mean(accuracy_values)

  print(f"Layers: {n_layers}, Average MRRMSE: {average_mrrmse}, Average accuracy: {average_acc}")

  if average_mrrmse < best_mrrmse:
    best_mrrmse = average_mrrmse
    best_layer_val = n_layers

print(f"Best Num Layers: {best_layer_val}")

test_loss, test_acc = model.evaluate([X_test[:, :n_celltypes], X_test[:, n_celltypes:]], y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
