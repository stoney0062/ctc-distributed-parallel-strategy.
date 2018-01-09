BEGIN{
	for(i=1 ; i<=10;i++){
		a[i] = 0	
	}
	casenum = 0	
}
{
	if($1=="Sequence"){
			start=NR	
			pinyinpos = 0
			topwords = ""
			topN = 0
			casenum += 1
	}	
	else if(NR==(start+1)){
			for (i=NF ; i>=1 ;i--){
					#if($i~/^[a-z]+$/){
					#print $i,substr($i,1,1)
					if(substr($i,1,1)=="*"){
							pinyinpos = i
							break
					}	
			}	
			for (i=pinyinpos+1 ; i<=NF ; i++){
					topwords = topwords""$i
			}
	}
	else{
			topN += 1
			toppos = 0
			decode = ""
			for (i=NF ; i>=1 ;i--){
					if($i=="Top"){
							toppos = i+2
							break
					}	
			}	
			for (i=toppos ; i<=NF ; i++){
					decode = decode""$i
			}
			if (decode == topwords){
			 		a[topN] += 1	
			}
	}
}
END{
		print "-------------------------------------"
		for( i=1 ; i<=length(a) ; i++){
				print "top",i,"--->",a[i],"times"
				count += a[i]
				if ( i == 3 )	top3 = count;
				if ( i == 5 )  top5 = count;
				if ( i==length(a) ) top10 = count;
		}	
		print "-------------------------------------"
		print "All case num:",casenum
		print "-------------------------------------"
		print "Top 1",a[1]/casenum
		print "Top 3",top3/casenum
		print "Top 5",top5/casenum
		print "Top 10",top10/casenum
		print "-------------------------------------"

}
