for num in 1 2 3 4 5
do
    python train-height-clap.py 0 >> 1.txt
    wait
    python train-height-pushpull.py 0 >> 2.txt
    wait
    python train-height-slide.py 0 >> 3.txt
    wait
    python train-height-sweep.py 0 >> 4.txt
    wait
done
