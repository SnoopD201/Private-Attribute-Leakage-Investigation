for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        python Gender/train-gender-no-${t}.py 1
        wait 
        python Gender/test-${t}.py >> Gender/${t}-test1.txt 
        wait
    done

    for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        python Height/train-height-no-${t}.py 1
        wait 
        python Height/test-${t}.py >> Height/${t}-test1.txt 
        wait
    done
