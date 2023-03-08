for i in SUC*.py; do python $i &> ${i}.txt | tee ${i}.txt; done
