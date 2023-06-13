for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 6 7 8 9 10
        do
                python train-height-no-${t}.py 0
                wait 
                python test-${t}.py >> ${t}-testexclude6.txt 
                wait
        done
done