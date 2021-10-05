import time

""" A simple class for estimating how long it takes to run a batch with similar size elements. """
class ExecutionTimeTracker():
    def markStartOfLoop(self,startIterationNumber = 0):
        self.startTime = time.time()
        self.startIterationNumber = startIterationNumber

    def markCompletedIteration(self, iterationsCompleted, totalIterations):
        self.latestIterationStopTime = time.time()
        self.iterationsCompleted = iterationsCompleted
        self.totalIterations = totalIterations
    
    def getExpectedTimeOfFinishingStr(self):
        timePerIteration = (self.latestIterationStopTime-self.startTime)/(self.iterationsCompleted-self.startIterationNumber)
        timeRemaining = (self.totalIterations-self.iterationsCompleted)*timePerIteration

        completionTime = self.latestIterationStopTime + timeRemaining

        return time.ctime(completionTime)

if __name__ == '__main__':
    # Do a simple test.

    numIterations = 10

    ett = ExecutionTimeTracker()

    ett.markStartOfLoop()
    print("start time={}".format(time.ctime(time.time())))

    for i in range(numIterations):
        time.sleep(1)
        
        ett.markCompletedIteration(i+1,numIterations)
        print("At iteration {}, expected time of completion is {}".format(i+1,ett.getExpectedTimeOfFinishingStr()))
