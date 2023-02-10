from config import CFG
import model_builder

s_bert = model_builder.load_sbert(model_path=CFG.CLASSIFIER_MODEL_PATH,
                                  sbert_path=CFG.SBERT_MODEL_FOLDER)
classifier = model_builder.get_classifier()
classifier.load_weights()