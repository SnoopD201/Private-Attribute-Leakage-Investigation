for num in 1 2 3 4 
do
    python train-weight-sweep.py 0 >> 4.txt
    wait
    python train-weight-clap.py 0 >> 1.txt
    wait
    python train-weight-pushpull.py 0 >> 2.txt
    wait
    python train-weight-slide.py 0 >> 3.txt
    wait
done
