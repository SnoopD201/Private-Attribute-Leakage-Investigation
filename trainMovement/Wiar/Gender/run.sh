for num in 1 2 3 4 5 6 7 8 9 10
do
    for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        do
            python train-gender-${t}.py 1 >> output-${t}.txt
        done
done

