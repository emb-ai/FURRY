#pragma once
#include <cstdio>
#include <cstdarg>
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_ERROR 6
inline int __android_log_print(int,const char*,const char* format,...){va_list args;va_start(args,format);int n=vfprintf(stderr,format,args);fputc('\n',stderr);va_end(args);return n;}
