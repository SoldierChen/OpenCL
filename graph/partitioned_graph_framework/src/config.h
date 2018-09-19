#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <climits>

#define PR
#define PROP_TYPE float
#define MAX_PROP (CHAR_MAX)
#define STATUS_TYPE char
#define BLK_SIZE (128*1024)

#ifdef BFS
inline PROP_TYPE compute(PROP_TYPE uProp, PROP_TYPE eProp, PROP_TYPE vProp){
	PROP_TYPE tmp = uProp + eProp;
	if(tmp < vProp) return tmp;
	else return vProp;
}

inline bool updateCondition(PROP_TYPE vProp, PROP_TYPE tProp){
	return (vProp > tProp);
}
#endif
#ifdef PR
inline PROP_TYPE compute(PROP_TYPE uProp, PROP_TYPE eProp, PROP_TYPE vProp){
	return (0.85*uProp + vProp);
}
inline bool updateCondition(PROP_TYPE vProp, PROP_TYPE tProp){
  return (vProp != tProp);
}


#elif SSSP
inline PROP_TYPE compute(PROP_TYPE uProp, PROP_TYPE eProp, PROP_TYPE vProp){
	PROP_TYPE tmp = uProp + eProp;
	if(tmp < vProp) return tmp;
	else return vProp;
}

inline bool updateCondition(PROP_TYPE vProp, PROP_TYPE tProp){
	return (vProp > tProp);
}


#elif CC
inline PROP_TYPE compute(PROP_TYPE uProp, PROP_TYPE eProp, PROP_TYPE vProp){
	PROP_TYPE tmp = uProp + eProp;
	if(tmp < vProp) return tmp;
	else return vProp;
}

inline bool updateCondition(PROP_TYPE vProp, PROP_TYPE tProp){
	return (vProp > tProp);
}

#endif

#endif

