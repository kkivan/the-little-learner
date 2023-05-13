#lang racket
(require malt)

(define remainder
  (λ (x y)
    (cond
      ((< x y) x)
      (else (remainder (- x y) y)))))

(= (remainder 13 4) 1)

(define line
  (λ (x)
    (λ (theta)
      (let ((w (ref theta 0))
        (b (ref theta 1)))
      (+ (* w x) b)))))

(define points '(1 2 3))

((line 1) (list 1 10))

(map line points)

(define line-xs
  (tensor 2.0 1.0 4.0 3.0))

(define line-ys
  (tensor 1.8 1.2 4.2 3.3))


((line 7.3) (list 1.0 0.0))

(rank (tensor (tensor 1)))


(define l2-loss
  (λ (target)
    (λ (xs ys)
      (λ (θ)
        (let ((pred-ys ((target xs) θ)))
          (sum
           (sqr
            (- ys pred-ys))))))))

(define obj ((l2-loss line) line-xs line-ys))

(obj (list 0.0 0.0))
(obj (list 0.0099 0.0))

(gradient-of sqr (tensor 27))

(gradient-of obj (list 0 0))

(define revise
  (λ (f revs θ)
    (cond
      ((zero? revs) θ)
      (else
       (revise f (sub1 revs) (f θ))))))

(define f
  (λ (θ)
    (map (λ (p)
           (- p 3))
         θ)))

(revise f 5 (list 2 3))

(declare-hyper alpha)
(declare-hyper revs)
(declare-hyper batch-size)

(define gradient-descent
  (λ (obj θ)
    (let ((f (λ (Θ)
               (map (λ (p g)
                      (- p (* alpha g)))
                    Θ
                    (∇ obj Θ)))))
      (revise f revs θ))))

(with-hypers ((alpha 0.01)
              (revs 1000))
  (gradient-descent obj (list 0.0 0.0)))


(define quad
  (λ (t)
    (λ (θ)
      (+ (* (tref θ 0) (sqr t))
         (+ (* (tref θ 1) t) (tref θ 2))))))

(define plane
  (λ (t)
    (λ (θ)
      (+ (• (ref θ 0) t) (ref θ 1)))))

(define •
  (λ (w t)
    (sum (* w t))))

(define plane-xs
  (tensor (tensor 1.0 2.05)
          (tensor 1.0 3.0)
          (tensor 2.0 2.0)
          (tensor 2.0 3.91)
          (tensor 3.0 6.13)
          (tensor 4.0 8.09)))

(define plane-ys
  (tensor 13.99
          15.99
          18.0
          22.4
          30.2
          37.94))

(time (with-hypers ((revs 1000)
              (alpha 0.001))
  (gradient-descent
   ((l2-loss plane) plane-xs plane-ys)
   (list (tensor 0.0 0.0) 5.78))))

((plane
  (tensor 2.0 3.91))
 (list (tensor 3.5824044661594274
                 2.1381545838037486)
         6.371957766427741))

(samples 20 3)


(define sampling-obj
  (λ (expectant xs ys)
    (let ((n (tlen xs)))
      (λ (θ)
        (let ((b (samples n batch-size)))
          ((expectant (trefs xs b) (trefs ys b)) θ))))))

(time (with-hypers
    ((revs 1000)
     (alpha 0.01)
     (batch-size 4))
  (gradient-descent
   (sampling-obj
    (l2-loss line) line-xs line-ys)
   (list 0.0 0.0))))

(time (with-hypers ((revs 15000)
              (alpha 0.001))
  (gradient-descent
   ((l2-loss plane) plane-xs plane-ys)
   (list (tensor 0.0 0.0) 0.0))))

(time
 (with-hypers
     ((revs 15000)
      (alpha 0.001)
      (batch-size 4))
   (gradient-descent
    (sampling-obj
     (l2-loss plane) plane-xs plane-ys)
    (list (tensor 0.0 0.0) 0.0))))

      
(define lonely-i
  (λ (θ)
    (map (λ (p)
           (list p))
         θ)))

(define lonely-d
  (λ (Θ)
    (map (λ (p)
           (ref p 0))
         Θ)))

(define lonely-u
  (λ (Θ gs)
    (map (λ (p g)
           (list (- (tref p 0) (* alpha g))))
         Θ
         gs)))


(lonely-d (lonely-i (list 1 2 3)))


         
            