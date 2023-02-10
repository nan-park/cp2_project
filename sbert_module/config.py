class CFG:
    RANDOM_STATE = 42
    BERT_BATCH_SIZE = 32
    BERT_EPOCHS = 1
    BATCH_SIZE = 32
    EPOCHS = 50
    
    
    MAX_LENGTH = 512
    N_CLASSES = 15
    PATIENCE = 10
    
    PRETRAINED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    DATA_PATH = '/content/gdrive/MyDrive/cp2/sbert_module/data'
    SBERT_CSV_FILE = 'review.csv'
    MODEL_CSV_FILE = 'review.csv'
    SBERT_MODEL_PATH = '/content/gdrive/MyDrive/cp2/sbert_module/sbert_model'
    SBERT_MODEL_FOLDER = '23-02-10_0752'
    CLASSIFIER_MODEL_PATH = '/content/gdrive/MyDrive/cp2/sbert_module/model'
    CLASSIFIER_MODEL_DATE = '23-02-10_0739'
    CLASSIFIER_MODEL_FILE = 'BASELINE'

    VECTORIZED_FILE = 'df_vectorized.npz'
    
    CLASS_NAMES = [
        '가성비','귀여운','넓은','단체','만족',
        '모던','분위기','비주얼','아늑','위생',
        '응대','이색음식','이색테마','클래식','혼자'
    ]
