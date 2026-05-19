! Build (from this directory):
!   gfortran -O2 -o rmathlib_bug_demo rmathlib_bug_demo.f90 refs/cdflib.f90
!
! rmathlib values below from rmathlib 1.1.0 via
!   pgamma(x, alph, 1.0, lower_tail, log_p)  (and, in case 2, .exp()).
!
! Any working CDF implementation must satisfy P(a, x) + Q(a, x) = 1.

program rmathlib_bug_demo
  implicit none
  integer, parameter :: rk = kind(1.0d0)
  external :: gamma_inc

  real(kind=rk) :: a, x, p, q, rm_p, rm_q
  integer :: ind

  write(*, '(a)') 'Case 1: pgamma(99.0, 100.0, 1.0, true, false)  (linear-space)'
  rm_p =  5415.993554165666_rk
  rm_q = -5414.993554165666_rk
  write(*, '(a)') &
    '                              P(a,x)                   Q(a,x)              P + Q'
  write(*, '(a, es25.17e3, a, es25.17e3, a, es25.17e3)') &
    '  rmathlib 1.1.0:    ', rm_p, '  ', rm_q, '  ', rm_p + rm_q
  a = 100.0_rk
  x = 99.0_rk
  ind = 0
  call gamma_inc(a, x, p, q, ind)
  write(*, '(a, es25.17e3, a, es25.17e3, a, es25.17e3)') &
    '  CDFLIB (a=100, x=99): ', p, '  ', q, '  ', p + q
  write(*, *)

  write(*, '(a)') 'Case 2: log-space then exp at a=0.5, x=0.001'
  write(*, '(a)') '  P = pgamma(0.001, 0.5, 1.0, true,  true).exp()'
  write(*, '(a)') '  Q = pgamma(0.001, 0.5, 1.0, false, true).exp()'
  rm_p = 0.035670591729679901_rk
  rm_q = 0.071353074052735332_rk
  write(*, '(a)') &
    '                              P(a,x)                   Q(a,x)              P + Q'
  write(*, '(a, es25.17e3, a, es25.17e3, a, es25.17e3)') &
    '  rmathlib 1.1.0:    ', rm_p, '  ', rm_q, '  ', rm_p + rm_q
  a = 0.5_rk
  x = 0.001_rk
  ind = 0
  call gamma_inc(a, x, p, q, ind)
  write(*, '(a, es25.17e3, a, es25.17e3, a, es25.17e3)') &
    '  CDFLIB (a=0.5, x=0.001): ', p, '  ', q, '  ', p + q
end program rmathlib_bug_demo
