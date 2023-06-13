for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 6 7 8 9 10
        do
                python train-gender-no-${t}.py 0 
                wait 
                python test-${t}.py >> ${t}-test-exclude6.txt 
                wait
        done        
done

# for num in 1 2 3 4 5 6 
# do
#     python train-gender-no-6.py 0 
#     wait 
#     python test-6.py >> 6-test.txt 
#     wait    
# done