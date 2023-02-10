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
    DATA_PATH = './data'
    SBERT_CSV_FILE = 'review.csv'
    MODEL_CSV_FILE = 'review.csv'
    MODEL_PATH = './model'
    CLASSIFIER_MODEL_FILE = 'BASELINE'
    SBERT_MODEL_FOLDER = 's_bert_230209_1'

    VECTORIZED_FILE = 'df_vectorized.npz'
    
    CLASS_NAMES = [
        '가성비','귀여운','넓은','단체','만족',
        '모던','분위기','비주얼','아늑','위생',
        '응대','이색음식','이색테마','클래식','혼자'
    ]