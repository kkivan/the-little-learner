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
      (let ((w (car theta))
        (b (car (cdr theta))))
      (+ (* w x) b)))))

(define points '(1 2 3))

((line 1) (list 1 10))
(map line points)

(define line-xs
  (tensor 2.0 1.0 4.0 3.0))

(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

(define theta (list 1.0 0.0))


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

(revise f 5 (list 1 2 3))

(define (descent)
  (let ((alpha 0.01))
    (let ((f (λ (theta)
               (let ((gs (gradient-of obj theta)))
                 (list (- (list-ref theta 0) (* alpha (list-ref gs 0)))
                       (- (list-ref theta 1) (* alpha (list-ref gs 1))))))))
      (revise f 1000 (list 0.0 0.0)))))



(descent)








