for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    do
        python Weight/train-weight-no-${t}.py 1
        wait 
        python Weight/test-${t}.py >> Weight/${t}-test.txt 
        wait
    done