! Reference tables for the gamma kernels: gamma_log, gamma, gamma_inc.

program gen_gamma_kernels
  implicit none
  integer, parameter :: rk = kind(1.0d0)
  real(kind=rk), external :: gamma_log, gamma_user
  external :: gamma_inc

  real(kind=rk), parameter :: as(*) = (/ &
    0.25_rk, 0.5_rk, 0.9_rk, 1.0_rk, 1.5_rk, 2.5_rk, 5.0_rk, &
    9.5_rk, 12.0_rk, 17.0_rk, 25.0_rk, 50.0_rk, 100.0_rk, 500.0_rk /)
  real(kind=rk), parameter :: log_special(*) = (/ &
    0.149999999999_rk, 0.15_rk, 0.150000000001_rk, &
    0.374999999999_rk, 0.375_rk, 0.375000000001_rk, &
    0.799999999999_rk, 0.8_rk, 0.800000000001_rk, &
    2.249999999999_rk, 2.25_rk, 2.250000000001_rk, &
    9.999999999999_rk, 10.0_rk, 10.000000000001_rk /)
  real(kind=rk), parameter :: gamma_special(*) = (/ &
    14.999999999999_rk, 15.0_rk, 15.000000000001_rk /)
  real(kind=rk), parameter :: inc_special_a(*) = (/ &
    0.149999999999_rk, 0.15_rk, 0.150000000001_rk, &
    0.374999999999_rk, 0.375_rk, 0.375000000001_rk, &
    0.799999999999_rk, 0.8_rk, 0.800000000001_rk, &
    2.249999999999_rk, 2.25_rk, 2.250000000001_rk, &
    9.999999999999_rk, 10.0_rk, 10.000000000001_rk, &
    19.999999999999_rk, 20.0_rk, 20.000000000001_rk /)
  real(kind=rk), parameter :: inc_special_x(*) = (/ &
    0.149999999999_rk, 0.15_rk, 0.150000000001_rk, &
    0.374999999999_rk, 0.375_rk, 0.375000000001_rk, &
    0.799999999999_rk, 0.8_rk, 0.800000000001_rk, &
    1.999999999999_rk, 2.0_rk, 2.000000000001_rk, &
    9.999999999999_rk, 10.0_rk, 10.000000000001_rk, &
    19.999999999999_rk, 20.0_rk, 20.000000000001_rk /)
  integer :: i, j, unit, ind
  real(kind=rk) :: a, x, p, q, max_x, step

  ! gamma_log
  open(newunit=unit, file='tests/data/gamma_log.csv', status='replace', action='write')
  write(unit, '(a)') '# a, ln_gamma(a)'
  a = 0.05_rk
  do while (a <= 20.0_rk + 1.0e-12_rk)
    call putval(unit, a, .false.)
    call putval(unit, gamma_log(a), .true.)
    a = a + 0.05_rk
  end do
  do i = 1, size(log_special)
    a = log_special(i)
    call putval(unit, a, .false.)
    call putval(unit, gamma_log(a), .true.)
  end do
  close(unit)

  ! gamma
  open(newunit=unit, file='tests/data/gamma.csv', status='replace', action='write')
  write(unit, '(a)') '# a, gamma(a)'
  a = 0.1_rk
  do while (a <= 14.0_rk + 1.0e-12_rk)
    call putval(unit, a, .false.)
    call putval(unit, gamma_user(a), .true.)
    a = a + 0.1_rk
  end do
  do i = 1, size(gamma_special)
    a = gamma_special(i)
    call putval(unit, a, .false.)
    call putval(unit, gamma_user(a), .true.)
  end do
  close(unit)

  ! gamma_inc
  open(newunit=unit, file='tests/data/gamma_inc.csv', status='replace', action='write')
  write(unit, '(a)') '# a, x, P(a, x), Q(a, x)'
  do i = 1, size(as)
    a = as(i)
    if (a > 1.0_rk) then
      max_x = 3.0_rk * a
    else
      max_x = 8.0_rk
    end if
    step = max_x / 50.0_rk
    x = step / 2.0_rk
    do while (x <= max_x + 1.0e-12_rk)
      ind = 0
      p = 0.0_rk
      q = 0.0_rk
      call gamma_inc(a, x, p, q, ind)
      if (p /= 2.0_rk) then  ! 2.0 is CDFLIB's error sentinel
        call putval(unit, a, .false.)
        call putval(unit, x, .false.)
        call putval(unit, p, .false.)
        call putval(unit, q, .true.)
      end if
      x = x + step
    end do
  end do
  do i = 1, size(inc_special_a)
    a = inc_special_a(i)
    do j = 1, size(inc_special_x)
      x = inc_special_x(j)
      ind = 0
      p = 0.0_rk
      q = 0.0_rk
      call gamma_inc(a, x, p, q, ind)
      if (p /= 2.0_rk) then
        call putval(unit, a, .false.)
        call putval(unit, x, .false.)
        call putval(unit, p, .false.)
        call putval(unit, q, .true.)
      end if
    end do
  end do
  close(unit)

  write(0, '(a)') 'wrote 3 tables under tests/data/'

contains
  subroutine putval(unit, v, last)
    integer, intent(in) :: unit
    real(kind=rk), intent(in) :: v
    logical, intent(in) :: last
    character(len=32) :: buf
    write(buf, '(es25.17e3)') v
    if (last) then
      write(unit, '(a)') trim(adjustl(buf))
    else
      write(unit, '(a, a)', advance='no') trim(adjustl(buf)), ','
    end if
  end subroutine putval
end program gen_gamma_kernels
