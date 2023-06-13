for p in 1 2 3 4
do
    for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        do
            python train-gender-no-${t}.py 0
            wait 
            python test-${t}.py >> ${t}-test.txt 
            wait
        done
done