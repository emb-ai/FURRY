#include "button_latch.h"
#include <stdexcept>
int main(){
 ButtonLatch y;
 auto check=[](bool ok){if(!ok)throw std::runtime_error("Recording button latch regression");};
 check(y.Update(true,true,0));
 for(int i=1;i<100;i++)check(!y.Update(true,true,i*.01));
 check(!y.Update(false,false,1));check(!y.Update(true,true,1.1));
 check(!y.Update(true,false,1.2));check(!y.Update(true,true,1.23));
 check(!y.Update(true,false,1.3));check(!y.Update(true,false,1.41));
 check(y.Update(true,true,1.5));check(!y.Update(true,true,1.6));
}
