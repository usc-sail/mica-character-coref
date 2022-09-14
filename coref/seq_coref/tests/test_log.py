from mica_text_coref.coref.seq_coref.tests import test_log_two
from absl import logging
from absl import app

def g(argv):
    formatter = logging.PythonFormatter(fmt="%(message)s")
    logging.get_absl_handler().use_absl_log_file()
    logging.get_absl_handler().setFormatter(fmt=formatter)
    logging.info("Test Log")
    test_log_two.f()

if __name__=="__main__":
    app.run(g)