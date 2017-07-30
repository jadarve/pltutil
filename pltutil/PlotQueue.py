'''
Created on 2 Jan 2015

@author: jadarve
'''


from multiprocessing import Pool
from threading import Semaphore
import signal


__all__ = ['PlotQueue']


class PlotQueue(object):

    def __init__(self, writePath, processCount=4, queueSize=10):

        self.writePath = writePath
        self.processCount = processCount
        self.plotCount = 0
        self.plotsCompleted = 0
        self.queueSize = queueSize
        self.processPool = Pool(self.processCount)
        self.queueSemaphore = Semaphore(self.queueSize)
        self.callbackList = list()
        
        signal.signal(signal.SIGINT, self.__signintHandler)

    
    def __del__(self):
        
        self.processPool.close()
        self.processPool.join()
        

    def enqueuePlot(self, desc):
        
        # aquire the semaphore before sending the job
        self.queueSemaphore.acquire()
        
        self.processPool.apply_async(_plotWorker,
            [desc, self.writePath.format(self.plotCount)],
            callback=self.__plotCallback)

        self.plotCount += 1
        
    
    def close(self):
        
        self.processPool.close()
    
    def join(self):
        
        self.processPool.join()



    def addCallback(self, callback):

        self.callbackList.append(callback)


    def __plotCallback(self, obj):

        # release the semaphore once the plotting has been completed
        self.queueSemaphore.release()
        self.plotsCompleted += 1

        for pc in self.callbackList:
            try:
                pc(self.plotsCompleted)
            except Exception as e:
                # ignore any error while calling the callback
                print('Error in callback function: {0}'.format(e))
                
                
    def __signintHandler(self, signal, frame):
        
        print('PlotQueue: SIGINT received, shutting down plotting pool.')
        
        # close the pool
        self.processPool.close()
#         self.processPool.join()
#         self.processPool.terminate()
        
        
        
        print('PlotQueue: process pool closed.')



def _plotWorker(desc, path):
    try:
        fig = plot(desc)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print('plot worker error: {0}'.format(e.errstr()))
        raise e
    
    except Warning:
        print('warning captured!')
