/* simple hash join on BRAM */
//----channel define----//
typedef struct shuffledData{
  uint num;
  uint idx;
  } shuffled_type;

#define ENDFLAG 0xffff
channel uint2 relR[8][16] __attribute__((depth(128)));
channel uint2 relS[8][16] __attribute__((depth(128)));
channel uint relRendFlagCh __attribute__((depth(8)));
channel uint gatherFlagCh __attribute__((depth(8)));

channel uint relSendFlagCh __attribute__((depth(128)));
channel uint2 buildCh[16] __attribute__((depth(512)));

#define HASH(K, MASK, SKIP) (((K) & MASK) >> SKIP)
#define RELR_L_NUM 1024*256*1
#define HASHTABLE_L_SIZE 1024*256*1
#define HASHTABLE_BUCKET_SIZE 4 // 2 byte per 
#define HASHTABLE_BUCKET_NUM HASHTABLE_L_SIZE/HASHTABLE_BUCKET_SIZE
//#define SW
__attribute__((task))
__kernel void relRead (
                        __global uint2 * restrict rTable, 
                        __global uint2 * restrict rTableReadRange, 
                        __global uint2 * restrict sTable, 
                        __global uint2 * restrict sTableReadRange
                      )
{
    uint rTableOffset = rTableReadRange[0].x;
    uint rTableReadNum = rTableReadRange[0].y;
    uint sTableOffset = sTableReadRange[0].x;
    uint sTableReadNum = sTableReadRange[0].y;
    //printf("sTableReadNum %d, \n", rTableReadNum);
  
 //   for(int i = rTableOffset; i < rTableOffset + rTableReadNum; i ++){
    for(int i = 0; i < rTableReadNum; i += 8){
      #pragma unroll 8
        for(int k = 0; k < 8; k ++){
          uint2 rtable_uint2 = rTable[i+k];
          switch((rtable_uint2.x) & 0xf){
            case 0 :  write_channel_altera(relR[k][0], rtable_uint2); break;
            case 1 :  write_channel_altera(relR[k][1], rtable_uint2); break;
            case 2 :  write_channel_altera(relR[k][2], rtable_uint2); break;
            case 3 :  write_channel_altera(relR[k][3], rtable_uint2); break;
            case 4 :  write_channel_altera(relR[k][4], rtable_uint2); break;
            case 5 :  write_channel_altera(relR[k][5], rtable_uint2); break;
            case 6 :  write_channel_altera(relR[k][6], rtable_uint2); break;
            case 7 :  write_channel_altera(relR[k][7], rtable_uint2); break;
            case 8 :  write_channel_altera(relR[k][8], rtable_uint2); break;
            case 9 :  write_channel_altera(relR[k][9], rtable_uint2); break;
            case 10 : write_channel_altera(relR[k][10], rtable_uint2); break;
            case 11 : write_channel_altera(relR[k][11], rtable_uint2); break;
            case 12 : write_channel_altera(relR[k][12], rtable_uint2); break;
            case 13 : write_channel_altera(relR[k][13], rtable_uint2); break;
            case 14 : write_channel_altera(relR[k][14], rtable_uint2); break;
            case 15 : write_channel_altera(relR[k][15], rtable_uint2); break;
          }
        }
      }
 

  write_channel_altera(relRendFlagCh, ENDFLAG);

    //for(int i = sTableOffset; i < (sTableOffset + sTableReadNum); i ++){
    for(int i = 0; i < (sTableReadNum); i += 8){
      #pragma unroll 8
        for(int k = 0; k < 8; k ++){
          uint2 stable_uint2 = sTable[i + k];
          switch((stable_uint2.x)& 0xf){
            case 0 :  write_channel_altera(relS[k][0], stable_uint2); break;
            case 1 :  write_channel_altera(relS[k][1], stable_uint2); break;
            case 2 :  write_channel_altera(relS[k][2], stable_uint2); break;
            case 3 :  write_channel_altera(relS[k][3], stable_uint2); break;
            case 4 :  write_channel_altera(relS[k][4], stable_uint2); break;
            case 5 :  write_channel_altera(relS[k][5], stable_uint2); break;
            case 6 :  write_channel_altera(relS[k][6], stable_uint2); break;
            case 7 :  write_channel_altera(relS[k][7], stable_uint2); break;
            case 8 :  write_channel_altera(relS[k][8], stable_uint2); break;
            case 9 :  write_channel_altera(relS[k][9], stable_uint2); break;
            case 10 : write_channel_altera(relS[k][10], stable_uint2); break;
            case 11 : write_channel_altera(relS[k][11], stable_uint2); break;
            case 12 : write_channel_altera(relS[k][12], stable_uint2); break;
            case 13 : write_channel_altera(relS[k][13], stable_uint2); break;
            case 14 : write_channel_altera(relS[k][14], stable_uint2); break;
            case 15 : write_channel_altera(relS[k][15], stable_uint2); break;
        }
      }
    }

  write_channel_altera(relSendFlagCh, ENDFLAG);

}
shuffled_type decoder(uchar opcode){
  uint idx;
  uint num;
  switch(opcode){
     case 0: idx = 0; num = 0; break;
     case 1: idx = 0; num = 1; break;
     case 2: idx = 1; num = 1; break;
     case 3: idx = 8; num = 2; break;
     case 4: idx = 2; num = 1; break;
     case 5: idx = 16; num = 2; break;
     case 6: idx = 17; num = 2; break;
     case 7: idx = 136; num = 3; break;
     case 8: idx = 3; num = 1; break;
     case 9: idx = 24; num = 2; break;
     case 10: idx = 25; num = 2; break;
     case 11: idx = 200; num = 3; break;
     case 12: idx = 26; num = 2; break;
     case 13: idx = 208; num = 3; break;
     case 14: idx = 209; num = 3; break;
     case 15: idx = 1672; num = 4; break;
     case 16: idx = 4; num = 1; break;
     case 17: idx = 32; num = 2; break;
     case 18: idx = 33; num = 2; break;
     case 19: idx = 264; num = 3; break;
     case 20: idx = 34; num = 2; break;
     case 21: idx = 272; num = 3; break;
     case 22: idx = 273; num = 3; break;
     case 23: idx = 2184; num = 4; break;
     case 24: idx = 35; num = 2; break;
     case 25: idx = 280; num = 3; break;
     case 26: idx = 281; num = 3; break;
     case 27: idx = 2248; num = 4; break;
     case 28: idx = 282; num = 3; break;
     case 29: idx = 2256; num = 4; break;
     case 30: idx = 2257; num = 4; break;
     case 31: idx = 18056; num = 5; break;
     case 32: idx = 5; num = 1; break;
     case 33: idx = 40; num = 2; break;
     case 34: idx = 41; num = 2; break;
     case 35: idx = 328; num = 3; break;
     case 36: idx = 42; num = 2; break;
     case 37: idx = 336; num = 3; break;
     case 38: idx = 337; num = 3; break;
     case 39: idx = 2696; num = 4; break;
     case 40: idx = 43; num = 2; break;
     case 41: idx = 344; num = 3; break;
     case 42: idx = 345; num = 3; break;
     case 43: idx = 2760; num = 4; break;
     case 44: idx = 346; num = 3; break;
     case 45: idx = 2768; num = 4; break;
     case 46: idx = 2769; num = 4; break;
     case 47: idx = 22152; num = 5; break;
     case 48: idx = 44; num = 2; break;
     case 49: idx = 352; num = 3; break;
     case 50: idx = 353; num = 3; break;
     case 51: idx = 2824; num = 4; break;
     case 52: idx = 354; num = 3; break;
     case 53: idx = 2832; num = 4; break;
     case 54: idx = 2833; num = 4; break;
     case 55: idx = 22664; num = 5; break;
     case 56: idx = 355; num = 3; break;
     case 57: idx = 2840; num = 4; break;
     case 58: idx = 2841; num = 4; break;
     case 59: idx = 22728; num = 5; break;
     case 60: idx = 2842; num = 4; break;
     case 61: idx = 22736; num = 5; break;
     case 62: idx = 22737; num = 5; break;
     case 63: idx = 181896; num = 6; break;
     case 64: idx = 6; num = 1; break;
     case 65: idx = 48; num = 2; break;
     case 66: idx = 49; num = 2; break;
     case 67: idx = 392; num = 3; break;
     case 68: idx = 50; num = 2; break;
     case 69: idx = 400; num = 3; break;
     case 70: idx = 401; num = 3; break;
     case 71: idx = 3208; num = 4; break;
     case 72: idx = 51; num = 2; break;
     case 73: idx = 408; num = 3; break;
     case 74: idx = 409; num = 3; break;
     case 75: idx = 3272; num = 4; break;
     case 76: idx = 410; num = 3; break;
     case 77: idx = 3280; num = 4; break;
     case 78: idx = 3281; num = 4; break;
     case 79: idx = 26248; num = 5; break;
     case 80: idx = 52; num = 2; break;
     case 81: idx = 416; num = 3; break;
     case 82: idx = 417; num = 3; break;
     case 83: idx = 3336; num = 4; break;
     case 84: idx = 418; num = 3; break;
     case 85: idx = 3344; num = 4; break;
     case 86: idx = 3345; num = 4; break;
     case 87: idx = 26760; num = 5; break;
     case 88: idx = 419; num = 3; break;
     case 89: idx = 3352; num = 4; break;
     case 90: idx = 3353; num = 4; break;
     case 91: idx = 26824; num = 5; break;
     case 92: idx = 3354; num = 4; break;
     case 93: idx = 26832; num = 5; break;
     case 94: idx = 26833; num = 5; break;
     case 95: idx = 214664; num = 6; break;
     case 96: idx = 53; num = 2; break;
     case 97: idx = 424; num = 3; break;
     case 98: idx = 425; num = 3; break;
     case 99: idx = 3400; num = 4; break;
     case 100: idx = 426; num = 3; break;
     case 101: idx = 3408; num = 4; break;
     case 102: idx = 3409; num = 4; break;
     case 103: idx = 27272; num = 5; break;
     case 104: idx = 427; num = 3; break;
     case 105: idx = 3416; num = 4; break;
     case 106: idx = 3417; num = 4; break;
     case 107: idx = 27336; num = 5; break;
     case 108: idx = 3418; num = 4; break;
     case 109: idx = 27344; num = 5; break;
     case 110: idx = 27345; num = 5; break;
     case 111: idx = 218760; num = 6; break;
     case 112: idx = 428; num = 3; break;
     case 113: idx = 3424; num = 4; break;
     case 114: idx = 3425; num = 4; break;
     case 115: idx = 27400; num = 5; break;
     case 116: idx = 3426; num = 4; break;
     case 117: idx = 27408; num = 5; break;
     case 118: idx = 27409; num = 5; break;
     case 119: idx = 219272; num = 6; break;
     case 120: idx = 3427; num = 4; break;
     case 121: idx = 27416; num = 5; break;
     case 122: idx = 27417; num = 5; break;
     case 123: idx = 219336; num = 6; break;
     case 124: idx = 27418; num = 5; break;
     case 125: idx = 219344; num = 6; break;
     case 126: idx = 219345; num = 6; break;
     case 127: idx = 1754760; num = 7; break;
     case 128: idx = 7; num = 1; break;
     case 129: idx = 56; num = 2; break;
     case 130: idx = 57; num = 2; break;
     case 131: idx = 456; num = 3; break;
     case 132: idx = 58; num = 2; break;
     case 133: idx = 464; num = 3; break;
     case 134: idx = 465; num = 3; break;
     case 135: idx = 3720; num = 4; break;
     case 136: idx = 59; num = 2; break;
     case 137: idx = 472; num = 3; break;
     case 138: idx = 473; num = 3; break;
     case 139: idx = 3784; num = 4; break;
     case 140: idx = 474; num = 3; break;
     case 141: idx = 3792; num = 4; break;
     case 142: idx = 3793; num = 4; break;
     case 143: idx = 30344; num = 5; break;
     case 144: idx = 60; num = 2; break;
     case 145: idx = 480; num = 3; break;
     case 146: idx = 481; num = 3; break;
     case 147: idx = 3848; num = 4; break;
     case 148: idx = 482; num = 3; break;
     case 149: idx = 3856; num = 4; break;
     case 150: idx = 3857; num = 4; break;
     case 151: idx = 30856; num = 5; break;
     case 152: idx = 483; num = 3; break;
     case 153: idx = 3864; num = 4; break;
     case 154: idx = 3865; num = 4; break;
     case 155: idx = 30920; num = 5; break;
     case 156: idx = 3866; num = 4; break;
     case 157: idx = 30928; num = 5; break;
     case 158: idx = 30929; num = 5; break;
     case 159: idx = 247432; num = 6; break;
     case 160: idx = 61; num = 2; break;
     case 161: idx = 488; num = 3; break;
     case 162: idx = 489; num = 3; break;
     case 163: idx = 3912; num = 4; break;
     case 164: idx = 490; num = 3; break;
     case 165: idx = 3920; num = 4; break;
     case 166: idx = 3921; num = 4; break;
     case 167: idx = 31368; num = 5; break;
     case 168: idx = 491; num = 3; break;
     case 169: idx = 3928; num = 4; break;
     case 170: idx = 3929; num = 4; break;
     case 171: idx = 31432; num = 5; break;
     case 172: idx = 3930; num = 4; break;
     case 173: idx = 31440; num = 5; break;
     case 174: idx = 31441; num = 5; break;
     case 175: idx = 251528; num = 6; break;
     case 176: idx = 492; num = 3; break;
     case 177: idx = 3936; num = 4; break;
     case 178: idx = 3937; num = 4; break;
     case 179: idx = 31496; num = 5; break;
     case 180: idx = 3938; num = 4; break;
     case 181: idx = 31504; num = 5; break;
     case 182: idx = 31505; num = 5; break;
     case 183: idx = 252040; num = 6; break;
     case 184: idx = 3939; num = 4; break;
     case 185: idx = 31512; num = 5; break;
     case 186: idx = 31513; num = 5; break;
     case 187: idx = 252104; num = 6; break;
     case 188: idx = 31514; num = 5; break;
     case 189: idx = 252112; num = 6; break;
     case 190: idx = 252113; num = 6; break;
     case 191: idx = 2016904; num = 7; break;
     case 192: idx = 62; num = 2; break;
     case 193: idx = 496; num = 3; break;
     case 194: idx = 497; num = 3; break;
     case 195: idx = 3976; num = 4; break;
     case 196: idx = 498; num = 3; break;
     case 197: idx = 3984; num = 4; break;
     case 198: idx = 3985; num = 4; break;
     case 199: idx = 31880; num = 5; break;
     case 200: idx = 499; num = 3; break;
     case 201: idx = 3992; num = 4; break;
     case 202: idx = 3993; num = 4; break;
     case 203: idx = 31944; num = 5; break;
     case 204: idx = 3994; num = 4; break;
     case 205: idx = 31952; num = 5; break;
     case 206: idx = 31953; num = 5; break;
     case 207: idx = 255624; num = 6; break;
     case 208: idx = 500; num = 3; break;
     case 209: idx = 4000; num = 4; break;
     case 210: idx = 4001; num = 4; break;
     case 211: idx = 32008; num = 5; break;
     case 212: idx = 4002; num = 4; break;
     case 213: idx = 32016; num = 5; break;
     case 214: idx = 32017; num = 5; break;
     case 215: idx = 256136; num = 6; break;
     case 216: idx = 4003; num = 4; break;
     case 217: idx = 32024; num = 5; break;
     case 218: idx = 32025; num = 5; break;
     case 219: idx = 256200; num = 6; break;
     case 220: idx = 32026; num = 5; break;
     case 221: idx = 256208; num = 6; break;
     case 222: idx = 256209; num = 6; break;
     case 223: idx = 2049672; num = 7; break;
     case 224: idx = 501; num = 3; break;
     case 225: idx = 4008; num = 4; break;
     case 226: idx = 4009; num = 4; break;
     case 227: idx = 32072; num = 5; break;
     case 228: idx = 4010; num = 4; break;
     case 229: idx = 32080; num = 5; break;
     case 230: idx = 32081; num = 5; break;
     case 231: idx = 256648; num = 6; break;
     case 232: idx = 4011; num = 4; break;
     case 233: idx = 32088; num = 5; break;
     case 234: idx = 32089; num = 5; break;
     case 235: idx = 256712; num = 6; break;
     case 236: idx = 32090; num = 5; break;
     case 237: idx = 256720; num = 6; break;
     case 238: idx = 256721; num = 6; break;
     case 239: idx = 2053768; num = 7; break;
     case 240: idx = 4012; num = 4; break;
     case 241: idx = 32096; num = 5; break;
     case 242: idx = 32097; num = 5; break;
     case 243: idx = 256776; num = 6; break;
     case 244: idx = 32098; num = 5; break;
     case 245: idx = 256784; num = 6; break;
     case 246: idx = 256785; num = 6; break;
     case 247: idx = 2054280; num = 7; break;
     case 248: idx = 32099; num = 5; break;
     case 249: idx = 256792; num = 6; break;
     case 250: idx = 256793; num = 6; break;
     case 251: idx = 2054344; num = 7; break;
     case 252: idx = 256794; num = 6; break;
     case 253: idx = 2054352; num = 7; break;
     case 254: idx = 2054353; num = 7; break;
     case 255: idx = 16434824; num = 8; break;
     default: idx = 0; num = 0; break;
 }
 shuffled_type data;
 data.idx = idx;
 data.num = num;
 return data;
}
__attribute__((task))
__kernel void gather (
                        __global uint * restrict matchedTable, 
                        __global uint2 * restrict rTableReadRange,  
                        __global uint2 * restrict sTableReadRange
                      )
{
    bool engine_finish[16];  
   #pragma unroll 16
    for(int j = 0; j < 16; j ++)
      engine_finish[j] = false;

    while(true){
      #pragma unroll 16
        for(int i = 0; i < 16; i ++){ 
        // each collect engine do their work
    
          uint2 data_r[8];
          bool valid_r[8];

          #pragma unroll 8
            for(int j = 0; j < 8; j ++){
              valid_r[j] = false;
            }
          
          #pragma unroll 8
            for(int j = 0; j < 8; j ++){  
              data_r[j] = read_channel_nb_altera(relR[j][i], &valid_r[j]);
            }
            // low is active
            engine_finish[i] = valid_r[0] | valid_r[1] | valid_r[2] | valid_r[3]| 
                               valid_r[4] | valid_r[5] | valid_r[6] | valid_r[7];


            uchar opcode = valid_r[0] + (valid_r[1] << 1) + (valid_r[2] << 2) + (valid_r[3] << 3) + (valid_r[4] << 4) + (valid_r[5] << 5) + (valid_r[6] << 6)
            + (valid_r[7] << 7);
            
            //printf("opcode 0X%2x ...", opcode);
            shuffled_type shuff_ifo = decoder(opcode);

            for(int j = 0; j < shuff_ifo.num; j ++){ 
              uint idx = (shuff_ifo.idx >> ((shuff_ifo.num - 1) * 3)) & 0x7;
              write_channel_altera(buildCh[i], data_r[idx]);
            }

        }

      // low is active
      bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
                        engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7] |
                        engine_finish[8] | engine_finish[9] | engine_finish[10]| engine_finish[11]| 
                        engine_finish[12]| engine_finish[13]| engine_finish[14]| engine_finish[15];

      bool valid_endflag = false;
      uint endFlagData;
      uint endFlag = read_channel_nb_altera (relRendFlagCh, &valid_endflag);
      if(valid_endflag ) endFlagData =  endFlag;
      if(endFlagData == ENDFLAG && !all_finish) 
      { 
        write_channel_altera(gatherFlagCh, ENDFLAG);
        break;
      } 
  }   
}

__attribute__((task))
__kernel void filter(){
    
}


__attribute__((task))
__kernel void hashjoin (
                        __global uint * restrict matchedTable, 
                        __global uint2 * restrict rTableReadRange,  
                        __global uint2 * restrict sTableReadRange
                      )
{
   // build phrase 
   //__local uint relR_l [RELR_L_SIZE];
    uint2 hashtable_l [HASHTABLE_L_SIZE >> 4][16];
    uint hashtable_bucket_cnt [HASHTABLE_BUCKET_NUM >> 4][16];
/*
    for(int i = 0; i < HASHTABLE_BUCKET_NUM; i ++){
      hashtable_bucket_cnt[i] = 0;
    }
    uint2 init_value;
    init_value.x = 0;
    init_value.y = 0;
    for(int i = 0; i < HASHTABLE_L_SIZE; i ++){
      hashtable_l[i] = init_value;
    }
*/
    uint rTableReadNum = rTableReadRange[0].y;
    bool engine_finish[16];  
   #pragma unroll 16
    for(int j = 0; j < 16; j ++)
      engine_finish[j] = false;

    while(true){
      #pragma unroll 16
        for(int i = 0; i < 16; i ++){ 
        // each collect engine do their work
            // low is active
          uint2 tmp_data = read_channel_nb_altera (buildCh[i], &engine_finish[i]);
          if(engine_finish[i]){
            uint key  = tmp_data.x; 
            uint val  = tmp_data.y;
            uint hash_idx = HASH (key,(HASHTABLE_BUCKET_NUM - 1),4);
            hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + hashtable_bucket_cnt[hash_idx][i]][i]= tmp_data;
            hashtable_bucket_cnt[hash_idx][i] ++; 
          }
        }
      // low is active
      bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
                        engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7] |
                        engine_finish[8] | engine_finish[9] | engine_finish[10]| engine_finish[11]| 
                        engine_finish[12]| engine_finish[13]| engine_finish[14]| engine_finish[15];

      bool valid_endflag = false;
      uint endFlagData;
      uint endFlag = read_channel_nb_altera (gatherFlagCh, &valid_endflag);
      if(valid_endflag ) endFlagData =  endFlag;
      if(endFlagData == ENDFLAG && !all_finish) break; 
    }



    //  =--------------------------------------probe phrase--------------------------------------------//
    uint matchedCnt[16] = {0}; 
    uint iter = (sTableReadRange[0].y);

    
    while(1){

       bool engine_finish[16] = {false};

    #pragma unroll 16
      for(int i = 0; i < 16; i ++){
      // each collect engine do their work 
        bool valid_s[8] = {false};
        uint2 data_s[8];
        #pragma unroll 8
          for(int j = 0; j < 8; j ++){  
              data_s[j] = read_channel_nb_altera(relS[j][i], &valid_s[j]);
          }

          engine_finish[i] = valid_s[0] | valid_s[1] | valid_s[2]  | valid_s[3] | 
                             valid_s[4] | valid_s[5] | valid_s[6]  | valid_s[7] ;
/*
        #pragma unroll
          for(int j = 0; j < 8; j ++){  
            if(valid_s[j]){
              uint key = data_s[j].x;
              uint hash_idx = HASH (key, (HASHTABLE_BUCKET_NUM - 1),4);
              //printf("probe key %d \n", key);
              for(int k = 0; k < HASHTABLE_BUCKET_SIZE; k ++){
                    //uint hashTable_val = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + j].y;
                  uint hashTable_key = hashtable_l[hash_idx * HASHTABLE_BUCKET_SIZE + k][i].x;
                  if(key == hashTable_key){
                //   matchedTable[matchedCnt] = key;
                //   matchedTable[matchedCnt + 1] = rtable_uint2.y;
                //   matchedTable[matchedCnt + 2] = hashTable_val;
                      matchedCnt[i] ++;
                  }
              }
            }
          }
  */     //--------------------------------------//
      }
      // low level is active
      bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
                  engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7] |
                  engine_finish[8] | engine_finish[9] | engine_finish[10]| engine_finish[11]| 
                  engine_finish[12]| engine_finish[13]| engine_finish[14]| engine_finish[15];

      bool valid_endflag = false;
      uint endFlagData;
      uint endFlag = read_channel_nb_altera (relSendFlagCh, &valid_endflag);
      if(valid_endflag ) endFlagData =  endFlag;
      if(endFlagData == ENDFLAG && !all_finish) break; 
    }
 
    uint total_cnt = 0;
  #pragma unroll 16
    for(int i = 0; i < 16; i ++){
        total_cnt += hashtable_l[1][i].x;

    }
    matchedTable[0] = total_cnt;


  #ifdef SW
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", sTable[i].x);
      printf ("\n");
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", rTable[i].x);
      printf ("\n");
        for (int i  = 0; i < 100; i ++)
      printf("%d \t", hashtable_l[i].x);
      printf ("\n");

  #endif
}
/*
#pragma unroll 8
    for ( int i = 0; i < RELR_L_SIZE; i ++){
        relR_l [i] = rTable [i];
    }
*/
#if 0

#define LOCAL_MEMORY_NUM_BITS 15
#define LOCAL_MEMORY_NUM      (1<<LOCAL_MEMORY_NUM_BITS)  //32<<10)
#define BucketNum 1024 * 1024 * 4
__attribute__((reqd_work_group_size(128,1,1)))
__attribute__((num_compute_units(1)))

__kernel void buildHashTable(__global uint * restrict rTableOnDevice, __global uint * restrict rHashTable,
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize,__global uint * restrict tHist, __global uint * restrict rHashCount)
{
	uint numWorkItems = get_global_size(0);
  uint tid          = get_global_id(0);
	uint lid          = get_local_id(0);
	uint lsize        = get_local_size(0);
	uint key, val, hash_lock, hash_index, count;
  uint iteration = 0;
  iteration = tid;
  while (tid < size)
	{
		//key = rTableOnDevice[offset * 2 + tid * 2 + 0];
		//val = rTableOnDevice[offset * 2 + tid * 2 + 1];
		uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + tid * 2 ]);
		key  = rtable_uint2.x; val  = rtable_uint2.y;
		      //2, update the corresponding bucket with index: hash_index.
		      hash_index = djb2_hash(key) & (rHashTableBucketNum-1); // real bucket to populate
	        //tHist[hash_index] ++ ;
	        tHist[get_global_id(0) * rHashTableBucketNum + hash_index] ++ ;
          //count = l_rHashCount[hash_index]++; //count = rHashCount[hash_index]; rHashCount[hash_index] = count + 1;
		      //rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 0] = key;	  //hash：
          //rHashTable[hash_index * hashBucketSize * 2 + count * 2 + 1] = val;      //hash
		tid += numWorkItems;
	}

    barrier(CLK_GLOBAL_MEM_FENCE);
// to caculate the elements nums in each buckets 
    tid = get_global_id(0);
  while (tid < rHashTableBucketNum){

    count = 0;
    for(int i = 0; i < numWorkItems; i ++){
       count += tHist[i * rHashTableBucketNum + tid];
    }
    rHashCount[tid] = count;

     tid += numWorkItems;
   }
// build the prefix-sum table 
  barrier(CLK_GLOBAL_MEM_FENCE);
  tid = get_global_id(0);
  for (int now_item = 1; now_item < (numWorkItems-1); now_item ++){
    iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[now_item * rHashTableBucketNum + iteration]  += tHist[(now_item-1) * rHashTableBucketNum + iteration];
      iteration += numWorkItems;
    }
  }
  // the prefix sum for the thread ID 0; 
    tid = get_global_id(0);
    iteration = tid;
    while (iteration < rHashTableBucketNum){
        tHist[127 * rHashTableBucketNum + iteration]= 0;
      iteration += numWorkItems;
    }

  barrier(CLK_GLOBAL_MEM_FENCE);
// write result to hash table 
  tid = get_global_id(0);
  iteration = tid;
  while (iteration < size){
        uint2 rtable_uint2 = *(__global uint2 *)(&rTableOnDevice[offset * 2 + iteration * 2 ]);
        key  = rtable_uint2.x; val  = rtable_uint2.y;
        hash_index = djb2_hash(key) & (rHashTableBucketNum-1);
        if(tid != 0){
        *(__global uint2 *)(&rHashTable[hash_index * hashBucketSize * 2 + tHist[(tid-1) * rHashTableBucketNum + hash_index] * 2] ) = rtable_uint2;
        tHist[(tid-1) * rHashTableBucketNum + hash_index] ++;
        }
        else {
          *(__global uint2 *)(&rHashTable[hash_index * hashBucketSize * 2 + tHist[127 * rHashTableBucketNum + hash_index] * 2] ) = rtable_uint2;
          tHist[127 * rHashTableBucketNum + hash_index] ++;
        }
        iteration += numWorkItems;
  }

}

__attribute__((num_compute_units(4)))
__kernel void probeHashTable(__global uint * restrict rHashTable, __global uint * restrict  sTableOnDevice, __global  uint * restrict matchedTable, 
	 const uint offset, const uint size, const uint rHashTableBucketNum, const uint hashBucketSize, __global uint * restrict rHashCount)
{
	  uint numWorkItems = get_global_size(0);
    uint tid          = get_global_id(0);
    uint lid          = get_local_id(0);
	  int block_id      = get_group_id(0);
 // int block_size    = get_num_groups(0);
	  uint key, val, hash, count, matchedNum;
	  __local int local_counter[128];
//	if (lid == 0)
//	 local_counter = 0;
	  local_counter[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
	while(tid < size)
	{
		key = sTableOnDevice[offset * 2 + tid * 2 + 0];
		val = sTableOnDevice[offset * 2 + tid * 2 + 1];

			//since hash value calculation consumes only tens ms, so GPU will finish it first
//		hash = djb2_hash(key,rHashTableBucketNum);

			hash = djb2_hash(key) &(rHashTableBucketNum-1);//sim_hash(key,rHashTableBucketNum);

			//find out matched tuples in hash table for R table
			count = 0;
			//int hashBucketRealSize = rHashCount[hash];
			while(count < hashBucketSize) //before optimization:  hashBucketRealSize
			{
//if(atomic_cmpxchg(&rHashTable[hash * hashBucketSize * 2 + count * 2 + 0],0,0) != 0  && atomic_cmpxchg(&rHashTable[hash * hashBucketSize * 2 + count * 2 + 0],0,0) == key)
				if(rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] != 0 && rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] == key)
				{
					local_counter[lid]++;//matchedNum = atomic_inc(&local_counter); //0 matchedTable[block_id]
//					matchedTable[3 + matchedNum * 3 + 0] = key;
//					matchedTable[3 + matchedNum * 3 + 1] = val;
//					matchedTable[3 + matchedNum * 3 + 2] = rHashTable[hash * hashBucketSize * 2 + count * 2 + 1];
					count++;
				}
				else if (rHashTable[hash * hashBucketSize * 2 + count * 2 + 0] == 0)
					break;
				else
					count++;
			}

		 tid += numWorkItems;
	}
     barrier(CLK_LOCAL_MEM_FENCE);
	 if ( (lid&3) == 0)
	   local_counter[lid] = local_counter[lid] + local_counter[lid+1] + local_counter[lid+2] + local_counter[lid+3];
     barrier(CLK_LOCAL_MEM_FENCE);
	 if ((lid&15) == 0)
	   local_counter[lid] = local_counter[lid] + local_counter[lid+4] + local_counter[lid+8] + local_counter[lid+12];
     barrier(CLK_LOCAL_MEM_FENCE);

    int counter_s = 0;

	 if (lid == 0)
	  {
	    counter_s = local_counter[0]  + local_counter[16] + local_counter[32] + local_counter[48] +
	               local_counter[64] + local_counter[80] + local_counter[96] + local_counter[112];
	    matchedTable[block_id] = counter_s;
	  }

//sum the local_counter[local_size];
//     if (lid == 0)
//	   matchedTable[block_id] = local_counter;
}
#endif
