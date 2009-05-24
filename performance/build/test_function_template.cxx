void $FUNCTION(void)
{
    BEGIN_TEST(__FUNCTION__);

    $DESCRIPTION

    try {
    /************ BEGIN INITIALIZATION SECTION ************/
    $INITIALIZE
    /************* END INITIALIZATION SECTION *************/
    
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
    
        float warmup_time;
        {
            cudaEventRecord(start, 0);
    
    /************ BEGIN TIMING SECTION ************/
    $TIME
    /************* END TIMING SECTION *************/
            
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
    
            float ms_elapsed;
            cudaEventElapsedTime(&ms_elapsed, start, end);
            warmup_time = ms_elapsed / float(1000);
        }
    
        // only verbose
        //std::cout << "warmup_time: " << warmup_time << " seconds" << std::endl;
    
        static const size_t NUM_TRIALS = 5;
        static const size_t MAX_ITERATIONS = 1000;
        static const float MAX_TEST_TIME = 0.5;  //TODO allow to be set by user
    
        size_t NUM_ITERATIONS;
        if (warmup_time == 0)
            NUM_ITERATIONS = MAX_ITERATIONS;
        else
            NUM_ITERATIONS = std::min(MAX_ITERATIONS, std::max( (size_t) 1, (size_t) (MAX_TEST_TIME / warmup_time)));
    
        float trial_times[NUM_TRIALS];
    
        for(size_t trial = 0; trial < NUM_TRIALS; trial++){
            cudaEventRecord(start, 0);
            for(size_t i = 0; i < NUM_ITERATIONS; i++){
                 
    /************ BEGIN TIMING SECTION ************/
    $TIME
    /************* END TIMING SECTION *************/
    
            }
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
    
            float ms_elapsed;
            cudaEventElapsedTime(&ms_elapsed, start, end);
            trial_times[trial] = ms_elapsed / (float(1000) * float(NUM_ITERATIONS));
        }
    
        // only verbose
        //for(size_t trial = 0; trial < NUM_TRIALS; trial++){
        //    std::cout << "trial[" << trial << "]  : " << trial_times[trial] << " seconds\n";
        //}
    
        float best_time = *std::min_element(trial_times, trial_times + NUM_TRIALS);
    
    /************ BEGIN FINALIZE SECTION ************/
    $FINALIZE
    /************* END FINALIZE SECTION *************/
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    
    
        cudaError_t error = cudaGetLastError();
        if(error){
            RECORD_TEST_FAILURE(cudaGetErrorString(error));
        } else {
            RECORD_TEST_SUCCESS();
        }

    }  // end try
    catch (std::bad_alloc) {
        RECORD_TEST_FAILURE("std::bad_alloc");
    }
    catch (thrusttest::UnitTestException e) {
        RECORD_TEST_FAILURE(e);
    }


    END_TEST();
}
