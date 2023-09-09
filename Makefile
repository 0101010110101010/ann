ann:test.c
	gcc test.c -o ann -O -ggdb3 -lm
test:test.c
	gcc test.c -o ann -O -ggdb3 -lm && ./ann
push:
	git push -u origin main
