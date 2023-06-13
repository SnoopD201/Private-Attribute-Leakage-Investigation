for num in 1 2 3 4 5
do
    python train-gender-clap.py 0 >> 1.txt
    wait
    python train-gender-pushpull.py 0 >> 2.txt
    wait
    python train-gender-slide.py 0 >> 3.txt
    wait
    python train-gender-sweep.py 0 >> 4.txt
    wait
done

