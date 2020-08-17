import numpy as np
import time
X =[0.5,2.5]
Y =[0.2,0.9]

def f(w,b,x):
  return 1.0/(1.0 + np.exp(-(w*x+b)))

def error(w,b):
  err= 0.0
  for x,y in zip(X,Y):
    fx = f(w,b,x)
    err += 0.5 * (fx-y)**2
  return err


def grad_b(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)

def grad_w(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*(fx)*(1-fx)*x

def do_gradient_descent():
  seconds = time.time()
  w,b,eta,max_epoch = 10,-10,1,100
  for i in range(max_epoch):
    dw=0
    db=0
    for x,y in zip(X,Y):
      dw += grad_w(w,b,x,y)
      db += grad_b(w,b,x,y)
    w = w - eta *dw
    b = b - eta * db
  print("Vanilla Gradient Descent : " ,error(w,b))  
  print("Seconds = ", time.time() - seconds)
def do_moumentum_grad():
    seconds = time.time()
    w,b,eta,max_epoch = 0,0,1,10
    gama = 0.1
    w_pre,b_pre = 0,0
    for i in range(max_epoch):
        dw,db = 0,0
        for (x,y) in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - ((gama * w_pre) +(eta * dw))
        b = b - ((gama * b_pre) +(eta * db))
        w_pre = w
        b_pre = b
    print("Momentum Gradient Descent : ",error(w,b))
    print("Seconds = ", time.time() - seconds)
    
    
def do_nesterov_gradient():
    seconds = time.time()
    w,b,eta,max_epoch = 0,0,1,10
    gama = 0.1
    w_prev,b_prev = 0,0
    
    for i in range(max_epoch):
        dw,db=0,0
        v_w=gama*w_prev  
        v_b=gama*b_prev  
        for x,y in zip(X, Y):
            dw += grad_w(w-v_w,b-v_b,x, y)
            db += grad_b(w-v_w,b-v_b,x, y)
        
        v_w=gama*w_prev+eta*dw
        v_b=gama*b_prev+eta*db
        w=w-v_w
        b=b- v_b
        w_prev=v_w
        b_prev=v_b
        
    print("Nesterov Gradient Descent : ",error(w,b))
    print("Seconds = ", time.time() - seconds)

do_gradient_descent()
print("------------------------------------------------------")
do_moumentum_grad()
print("------------------------------------------------------")
do_nesterov_gradient()