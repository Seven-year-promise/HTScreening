#from modAL.models import ActiveLearner
#from modAL.uncertainty import uncertainty_sampling
#from sklearn.ensemble import RandomForestClassifier
from data_loader import load_cleaned_data, generate_training_set

def train(data, compounds, labels, output_dir):
    x_training, y_training, x_unlabelled = generate_training_set(data, labels)
    """
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=uncertainty_sampling,
    X_training = x_training, y_training = y_training
    )
    """
    #query_idx, query_sample = learner.query(x_unlabelled)

    # ...obtaining new labels from the Oracle...

    #learner.teach(query_sample, query_label)

if __name__ == "__main__":
    data, compounds, labels = load_cleaned_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned/all_compounds_ori_fish_with_action.csv")
    #print(data.shape, compounds.shape, labels.shape)
    train(data, compounds, labels, output_dir="")