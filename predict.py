import numpy as np

def predict_de(data):

    ct = data['cell_type']
    sm = data['small_molecule']

    cell_type_input = []
    for name in cell_types:
        if name == ct:
            cell_type_input.append(1)
        else:
            cell_type_input.append(0)


    reshaped_data1 = np.expand_dims(np.array(cell_type_input), axis=0)

    sm_input = []
    for name in small_molecules:
        if name == sm:
            sm_input.append(1)
        else:
            sm_input.append(0)

    reshaped_data2 = np.expand_dims(np.array(sm_input), axis=0)

    if sum(cell_type_input) == 0 and sum(sm_input) == 0:
      print('*** warning: neither input seen in training - results not likely to be accurate')

    predictor = [reshaped_data1, reshaped_data2]
    return final_model.predict(predictor)[0]


### EXAMPLE ###
trial = {
    'cell_type': 'NK cells',
    'small_molecule': 'Tivantinib'
}

pred = predict_de(trial)
print(f"== predictions for ==\ncell type: {trial['cell_type']}\nsmall molecule: {trial['small_molecule']}")
print(f'num genes predicted:\t{len(pred)}\n{pred}')

"""
1/1 [==============================] - 0s 76ms/step
== predictions for ==
cell type: NK cells
small molecule: Tivantinib
num genes predicted:	18211
[ 0.1258797   0.09605698  0.11574764 ...  0.12275812 -0.07510065
 -0.08342823]
"""
