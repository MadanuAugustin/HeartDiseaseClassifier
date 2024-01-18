import sys
from src.logger import logging

class UserException(Exception):
    def __init__(self, errormsg, system_error:sys):
        super().__init__(errormsg)
        self.errormsg = errormsg_function(errormsg = errormsg, system_error = system_error)


    def __str__(self):
        return self.errormsg
    

def errormsg_function(errormsg, system_error:sys):
    _,_,exc_tb = system_error.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    errormsg = "Error has been raised in the filename of {} and the linenbr is {} errormsg : {}".format(filename, lineno, str(errormsg))
    logging.info('Exception file has been executed...!')
    return errormsg
