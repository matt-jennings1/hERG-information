all: dns

dns:
	@echo "Building HH model"
	gcc -I/usr/local/SUNDIALS-2.4.0/include -L/usr/local/SUNDIALS-2.4.0/lib HHFullOut.c -o HHFullOut -lm -lsundials_cvode -lsundials_nvecserial 

cleandata:
	@echo "Cleaning data files only ..."
	rm -rf *.txt

clean:
	@echo "Cleaning up everything ... "
	rm -rf *.o *.out hh 
