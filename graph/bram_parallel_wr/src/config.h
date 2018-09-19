#ifndef __CONFIG_H__
#define __CONFIG_H__

#define PROP_TYPE int
#define MAX_PROP (256)
#define STATUS_TYPE char

// BFS kernel
inline int compute(PROP_TYPE uProp, PROP_TYPE eProp, PROP_TYPE vProp){
	PROP_TYPE tmp = uProp + eProp;
	if(tmp < vProp) return tmp;
	else return vProp;
}

#endif

