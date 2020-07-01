import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn.manifold import TSNE
from neural_network.Dataset import FullDataset
from neural_network.SNN import initialise_snn

from configuration.Configuration import Configuration

# In progress. For visualization of encoded data of an SNN using T-SNE or PCA.
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Config relevant for cnnwithclassattention
    use_channelwiseEncoded_for_cnnwithclassattention = True
    use_each_sensor_as_single_example = True  # default: false

    use_channelcontextEncoded_for_cnnwithclassattention = True

    # TSNE Embeddings are learned on a sim matrix, not on embeddings itself
    use_sim_matrix = False  # default false

    encode_test_data = False

    config = Configuration()
    config.architecture_variant = config.architecture_variants[0]
    if config.case_base_for_inference:
        dataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset = FullDataset(config.training_data_folder, config, training=False)
    # config.case_base_folder
    dataset.load()
    print("130:", dataset.y_train_strings[130])
    print("200:", dataset.y_train_strings[200])
    print("700:", dataset.y_train_strings[700])
    print("750:", dataset.y_train_strings[750])
    print("650:", dataset.y_train_strings[650])
    print("550:", dataset.y_train_strings[550])
    print("790:", dataset.y_train_strings[790])
    print("730:", dataset.y_train_strings[730])

    architecture = initialise_snn(config, dataset, False)

    if architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
        if config.use_same_feature_weights_for_unsimilar_pairs == True:
            print("config.use_same_feature_weights_for_unsimilar_pairs should be False during TSNE Plot!")

    # As TSNE Input either a distance matrix or
    if use_sim_matrix == True:
        sim_matrix = dataset.get_similarity_matrix(architecture, encode_test_data=encode_test_data)
        distance_matrix = 1 - sim_matrix
    else:
        encoded_data = dataset.encode(architecture, encode_test_data=encode_test_data)
    if architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
        x_train_encoded = dataset.x_train[0]
        x_train_encoded_context = np.squeeze(dataset.x_train[2])
        x_train_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings
    else:
        # Loading encoded data previously created by the DatasetEncoder.py
        x_train_encoded = dataset.x_train
        x_test_encoded = dataset.x_test
        x_train_labels = dataset.y_train_strings
        x_test_labels = dataset.y_test_strings

    if encode_test_data:
        print("Loaded encoded data: ", x_train_encoded.shape, " ", x_test_encoded.shape)
    else:
        print("Loaded encoded data: ", x_train_encoded.shape)

    # Encoding / renaming of labels from string value (e.g. no error, ....) to integer (e.g. 0)
    le = preprocessing.LabelEncoder()
    if use_each_sensor_as_single_example:
        # Generate new labels
        x_test_train_labels = np.tile(dataset.feature_names_all, x_train_encoded.shape[0])
        le.fit(dataset.feature_names_all)
        numOfClasses = le.classes_.size
        unique_labels_EncodedAsNumber = le.transform(le.classes_)  # each label encoded as number
        x_trainTest_labels_EncodedAsNumber = le.transform(x_test_train_labels)
    else:
        x_test_train_labels = np.concatenate((x_train_labels, x_test_labels),
                                             axis=0)
        le.fit(x_test_train_labels)
        numOfClasses = le.classes_.size
        # print("Number of classes detected: ", numOfClasses, ". \nAll classes: ", le.classes_)
        unique_labels_EncodedAsNumber = le.transform(le.classes_)  # each label encoded as number
        x_trainTest_labels_EncodedAsNumber = le.transform(x_train_labels)

    # Converting / reshaping 3d encoded features to 2d (required as TSNE/PCA input)
    data4TSNE = None
    if architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
        if use_sim_matrix == False:
            if use_channelwiseEncoded_for_cnnwithclassattention:
                num_of_flatten_features = x_train_encoded.shape[1] * x_train_encoded.shape[2]
            if use_channelcontextEncoded_for_cnnwithclassattention:
                num_of_flatten_features = x_train_encoded_context.shape[1]
            if use_channelcontextEncoded_for_cnnwithclassattention & use_channelwiseEncoded_for_cnnwithclassattention:
                num_of_flatten_features = x_train_encoded.shape[1] * x_train_encoded.shape[2] + \
                                          x_train_encoded_context.shape[1]

            if use_each_sensor_as_single_example:
                data4TSNE = np.zeros((x_train_encoded.shape[0] * x_train_encoded.shape[2], x_train_encoded.shape[1]))
            else:
                data4TSNE = np.zeros((x_train_encoded.shape[0], num_of_flatten_features))

            for i in range(x_train_encoded.shape[0]):
                if use_channelwiseEncoded_for_cnnwithclassattention:
                    x = np.reshape(x_train_encoded[i, :, :], (x_train_encoded.shape[1] * x_train_encoded.shape[2]), 1)
                    # print("x: ", x.shape)
                if use_channelcontextEncoded_for_cnnwithclassattention:
                    y = np.squeeze(np.reshape(x_train_encoded_context[i, :], x_train_encoded_context.shape[1]))
                # print("x_train_encoded[2][i]: ", x_train_encoded_context[2][i].shape)
                if use_channelcontextEncoded_for_cnnwithclassattention & use_channelwiseEncoded_for_cnnwithclassattention:
                    x = np.concatenate((x, y))
                elif use_channelcontextEncoded_for_cnnwithclassattention:
                    x = y
                # print("x: ", x.shape)
                if use_each_sensor_as_single_example:
                    for s in range(x_train_encoded.shape[2]):
                        x = np.reshape(x_train_encoded[i, :, s], (x_train_encoded.shape[1]), 1)
                        data4TSNE[((i - 1) * 61 + s), :] = x
                else:
                    data4TSNE[i, :] = x
        else:
            data4TSNE = distance_matrix
        print("data4TSNE:", data4TSNE.shape)
    else:
        num_of_flatten_features = x_train_encoded.shape[1] * x_train_encoded.shape[2]
        data4TSNE = np.zeros((x_train_encoded.shape[0], num_of_flatten_features))
        for i in range(x_train_encoded.shape[0]):
            x = np.reshape(x_train_encoded[i, :, :], (x_train_encoded.shape[1] * x_train_encoded.shape[2]), 1)
            data4TSNE[i, :] = x
    '''
    x_train_encoded_reshapedAs2d = x_train_encoded.reshape(
        [x_train_encoded.shape[0], x_train_encoded.shape[1] * x_train_encoded.shape[2]])
    x_test_encoded_reshapedAs2d = x_test_encoded.reshape(
        [x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2]])
    print("Reshaped encoded data shape train: ", x_train_encoded_reshapedAs2d.shape, ", test: ",
          x_test_encoded_reshapedAs2d.shape)
    '''
    # Concatenate train and test data into one matrix
    # x_testTrain_encoded_reshapedAs2d = np.concatenate((x_train_encoded_reshapedAs2d, x_test_encoded_reshapedAs2d),
    #                                                  axis=0)

    # Reducing dimensionality with TSNE or PCA
    # metrics: manhattan, euclidean, cosine
    if architecture.hyper.encoder_variant in ['cnnwithclassattention', 'cnn1dwithclassattention']:
        if use_sim_matrix:
            X_embedded = TSNE(n_components=2, perplexity=50.0, learning_rate=10, early_exaggeration=30, n_iter=10000,
                              random_state=123, metric="precomputed").fit_transform(data4TSNE)
        else:
            X_embedded = TSNE(n_components=2, perplexity=50.0, learning_rate=10, early_exaggeration=30, n_iter=10000,
                              random_state=123, metric='manhattan').fit_transform(data4TSNE)
    else:
        X_embedded = TSNE(n_components=2, perplexity=50.0, learning_rate=10, early_exaggeration=10, n_iter=10000,
                          random_state=123, metric='manhattan').fit_transform(data4TSNE)
    # X_embedded = TSNE(n_components=2, random_state=123).fit_transform(data4TSNE)
    # X_embedded = PCA(n_components=2, random_state=123).fit_transform(data4TSNE)

    # dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    # file_name = '../data/visualizations/' + "reducedTestFeaturesViz.npy"
    # np.save(file_name, X_embedded)
    # X_embedded = np.load(file_name).astype('float32')
    print("X_embedded shape: ", X_embedded.shape)
    # print("X_embedded:", X_embedded[0:10,:])
    # Defining the color for each class

    colors = [plt.cm.jet(float(i) / max(unique_labels_EncodedAsNumber)) for i in range(numOfClasses)]
    # Color maps: https://matplotlib.org/examples/color/colormaps_reference.html
    # colors_ = colors(np.array(unique_labels_EncodedAsNumber))
    # Overriding color map with own colors
    colors[0] = np.array([0 / 256, 128 / 256, 0 / 256, 1])  # no failure
    '''
    colors[0] = np.array([0 / 256, 128 / 256, 0 / 256, 1])  # no failure
    colors[1] = np.array([65 / 256, 105 / 256, 225 / 256, 1])  # txt15_m1_t1_high_wear
    colors[2] = np.array([135 / 256, 206 / 256, 250 / 256, 1])  # txt15_m1_t1_low_wear
    colors[3] = np.array([123 / 256, 104 / 256, 238 / 256, 1])  # txt15_m1_t2_wear
    colors[4] = np.array([189 / 256, 183 / 256, 107 / 256, 1])  # txt16_i4
    colors[5] = np.array([218 / 256, 112 / 256, 214 / 256, 1])  # txt16_m3_t1_high_wear
    colors[6] = np.array([216 / 256, 191 / 256, 216 / 256, 1])  # txt16_m3_t1_low_wear
    colors[7] = np.array([128 / 256, 0 / 256, 128 / 256, 1])  # txt16_m3_t2_wear
    colors[8] = np.array([255 / 256, 127 / 256, 80 / 256, 1])  # txt_17_comp_leak
    colors[9] = np.array([255 / 256, 99 / 256, 71 / 256, 1])  # txt_18_comp_leak
    '''
    # Generating the plot
    rowCounter = 0

    for i, u in enumerate(unique_labels_EncodedAsNumber):
        # print("i: ",i,"u: ",u)
        for j in range(X_embedded.shape[0]):
            if x_trainTest_labels_EncodedAsNumber[j] == u:
                xi = X_embedded[j, 0]
                yi = X_embedded[j, 1]
                # print("i: ", i, " u:", u, "j:",j,"xi: ", xi, "yi: ", yi)
                plt.scatter(xi, yi, color=colors[i], label=unique_labels_EncodedAsNumber[i], marker='.')

    # print("X_embedded:", X_embedded.shape)
    # print(X_embedded)
    # print("x_trainTest_labels_EncodedAsNumber: ", x_trainTest_labels_EncodedAsNumber)
    plt.title("Visualization Train(.) and Test (x) data (T-SNE-Reduced)")
    lgd = plt.legend(labels=le.classes_, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=3)
    # plt.legend(labels=x_test_train_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # lgd = plt.legend(labels=le.classes_)
    for i, u in enumerate(le.classes_):
        lgd.legendHandles[i].set_color(colors[i])
        lgd.legendHandles[i].set_label(le.classes_[i])
    # plt.show()
    plt.savefig(architecture.hyper.encoder_variant + '_' + config.filename_model_to_use + '_730.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
