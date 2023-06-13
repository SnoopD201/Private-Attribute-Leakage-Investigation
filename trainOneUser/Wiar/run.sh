for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 6 7 8 9 10
        do
                python Weight/train-weight-no-${t}.py 0
                wait 
                python Weight/test-${t}.py >> Weight/${t}-test-exclude4.txt 
                wait
        done        
done

for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 6 7 8 9 10
        do
                python Height/train-height-no-${t}.py 0
                wait 
                python Height/test-${t}.py >> Height/${t}-test-exclude4.txt 
                wait
        done        
done

for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 6 7 8 9 10
        do
                python Gender/train-gender-no-${t}.py 0 
                wait 
                python Gender/test-${t}.py >> Gender/${t}-test-exclude4.txt 
                wait
        done        
done

# for num in 1 2 3 4 5 
# do
#     python train-gender-no-1.py 0 
#     wait 
#     python test-1.py >> 1-test.txt 
#     wait    
# done