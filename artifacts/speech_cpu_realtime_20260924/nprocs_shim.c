// LD_PRELOAD shim: make get_nprocs()/get_nprocs_conf() (and hence
// std::thread::hardware_concurrency) return SHIM_NPROCS. Used to control
// qwentts.cpp's CPU thread count (it hardcodes hardware_concurrency()/2).
#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>
static int val(void){ const char*s=getenv("SHIM_NPROCS"); return s?atoi(s):-1; }
int get_nprocs(void){ int v=val(); if(v>0) return v; int(*f)(void)=dlsym(RTLD_NEXT,"get_nprocs"); return f();}
int get_nprocs_conf(void){ int v=val(); if(v>0) return v; int(*f)(void)=dlsym(RTLD_NEXT,"get_nprocs_conf"); return f();}
