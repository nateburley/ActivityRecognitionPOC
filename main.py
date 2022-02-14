"""
Main function that trains a model to classify activities based on smart watch data.

Author: Nate Burley
"""
import data_module as data_mod
import training_module as train_mod
import models  # Right now redundant, but will be useful if we add more models in the future
import pickle

TRAIN_NEW = True


if __name__ == '__main__':
    # Read segmented raw data in
    # X_data, y_data = data_mod.load_raw_dataset()

    # Read pre-processed data in
    X_data, y_data =  data_mod.load_preprocessed_dataset('data/wisdm_preprocessed_watch_accel.csv')

    # Build training and testing sets
    X_train, X_test, y_train, y_test = data_mod.train_test_split(X_data,y_data,train_size=0.8, test_size=0.2, shuffle=True, stratify=y_data)

    ### DEBUGGING Print training and testing shape
    print(f"Train x shape: {X_train.shape}, y shape: {y_train.shape}")
    
    if TRAIN_NEW:
        print("Training from scratch... ")

        ## EXTRA TREES
        # Construct Extra Trees Classifier
        model = models.build_extra_trees_classifier()

        # Train the Extra Trees Classifier, (optionally uses grid search to find optimal parameters)
        model = train_mod.train_extra_trees(model, X_train, y_train, grid_search=True)


    else:
        print("Loading model...")

        # Open pickle file with existing model
        model_file = open('saved_models/best_extra_trees.pickle', 'rb')

        # Load model
        model = pickle.load(model_file)

    # Lastly, evaluate the model
    train_mod.evaluate_model(model, X_test, y_test, save=True)