! Reproduce the four originally-broken Rust doctests against the
! original Fortran cdflib. If the Fortran library produces the same
! outcome (same value, or same SearchOutOfBounds), the failure was
! inherited from CDFLIB itself. If Fortran succeeds where the Rust
! port fails, it's a port bug.
!
! Build (from this directory):
!   gfortran -O2 -o doctest_failures_demo doctest_failures_demo.f90 refs/cdflib.f90
!
! cdflib status codes:
!    0 = success
!   -i = the i-th input is out of range
!    1 = search did not converge
!    2 = search hit the lower bracket bound
!    3 = search hit the upper bracket bound
!    4 = P + Q != 1
!   10 = a numerical problem occurred

program doctest_failures_demo
  implicit none
  integer, parameter :: rk = kind(1.0d0)
  external :: gamma_inc, cdff, cdfchn, cdffnc

  call case1_gamma_inc()
  call case2_cdff_solve_dfn()
  call case3_cdfchn_solve_pnonc()
  call case4_cdffnc_solve_phonc()

contains

  subroutine case1_gamma_inc()
    real(kind=rk) :: a, x, p, q
    integer :: ind
    a = 2.5_rk; x = 1.7_rk; ind = 0
    call gamma_inc(a, x, p, q, ind)
    write(*, '(a)') '#1  gamma_inc(2.5, 1.7)'
    write(*, '(a, es24.16e3, a, es24.16e3)') '    Rust port:           P = ', 0.36143008_rk, ', Q = ', 0.63856992_rk
    write(*, '(a, es24.16e3, a, es24.16e3)') '    Fortran cdflib:      P = ', p, ', Q = ', q
    write(*, '(a)') '    original doctest:    P = 0.36135041, Q = 0.63864958  <-- wrong'
    write(*, *)
  end subroutine case1_gamma_inc

  subroutine case2_cdff_solve_dfn()
    integer :: which, status
    real(kind=rk) :: p, q, f, dfn, dfd, bound
    which = 3; p = 0.95_rk; q = 0.05_rk; f = 3.33_rk; dfn = 0.0_rk; dfd = 10.0_rk
    call cdff(which, p, q, f, dfn, dfd, status, bound)
    write(*, '(a)') '#2  FisherSnedecor::solve_dfn(0.95, 3.33, 10.0)'
    write(*, '(a)') '    Rust port: was Err(SearchOutOfBounds); now matches.'
    write(*, '(a, i0, a, es25.17e3, a, es25.17e3)') &
      '    Fortran cdff(which=3): status = ', status, ', dfn = ', dfn, ', bound = ', bound
    write(*, *)
  end subroutine case2_cdff_solve_dfn

  subroutine case3_cdfchn_solve_pnonc()
    integer :: which, status
    real(kind=rk) :: p, q, x, df, pnonc, bound
    which = 4; p = 0.5_rk; q = 0.5_rk; x = 15.0_rk; df = 5.0_rk; pnonc = 0.0_rk
    call cdfchn(which, p, q, x, df, pnonc, status, bound)
    write(*, '(a)') '#3  ChiSquaredNoncentral::solve_ncp(0.5, 15.0, 5.0)'
    write(*, '(a)') '    Rust port: was Err(SearchOutOfBounds); now matches.'
    write(*, '(a, i0, a, es25.17e3, a, es25.17e3)') &
      '    Fortran cdfchn(which=4): status = ', status, ', pnonc = ', pnonc, ', bound = ', bound
    write(*, *)
  end subroutine case3_cdfchn_solve_pnonc

  subroutine case4_cdffnc_solve_phonc()
    integer :: which, status
    real(kind=rk) :: p, q, f, dfn, dfd, phonc, bound
    which = 5; p = 0.5_rk; q = 0.5_rk; f = 4.0_rk; dfn = 5.0_rk; dfd = 10.0_rk; phonc = 0.0_rk
    call cdffnc(which, p, q, f, dfn, dfd, phonc, status, bound)
    write(*, '(a)') '#4  FisherSnedecorNoncentral::solve_ncp(0.5, 4.0, 5.0, 10.0)'
    write(*, '(a)') '    Rust port: was Err(SearchOutOfBounds); now matches.'
    write(*, '(a, i0, a, es25.17e3, a, es25.17e3)') &
      '    Fortran cdffnc(which=5): status = ', status, ', phonc = ', phonc, ', bound = ', bound
    write(*, *)
  end subroutine case4_cdffnc_solve_phonc

end program doctest_failures_demo
