import sys
import numerapi
from decouple import config
from numerai.definitions import PREDICTIONS_CSV

napi = numerapi.NumerAPI(config("public_id"), config("secret_key"))

napi.upload_predictions(PREDICTIONS_CSV, model_id=config(sys.argv[1]))
