#include<stdio.h>
#include<string.h>
int main()
{
	FILE *fp;
	fp = fopen("test1.txt","w");  
	fprintf(fp,"5\n");  
	fprintf(fp,"learn finished!\n"); 
	int i=1; 
	fprintf(fp,"%d\n",i);  
 } 
