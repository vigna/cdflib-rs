! Reference tables for every cdf* dispatcher in cdflib.f90.
!
! For each (dispatcher, which) branch, generate a CSV whose last column
! is the Fortran-computed output and the earlier columns are the inputs.
! Companion Rust test: tests/dispatchers.rs.

program gen_dispatchers
  implicit none
  integer, parameter :: rk = kind(1.0d0)
  integer :: trace_unit
  external :: cdfbet, cdfbin, cdfchi, cdfchn, cdff, cdffnc, &
              cdfgam, cdfnbn, cdfnor, cdfpoi, cdft
  external :: solver_trace_enable, solver_trace_disable

  open(newunit=trace_unit, file='tests/data/solver_traces.txt', status='replace', action='write')

  call gen_cdfbet()
  call gen_cdfbin()
  call gen_cdfchi()
  call gen_cdfchn()
  call gen_cdff()
  call gen_cdffnc()
  call gen_cdfgam()
  call gen_cdfnbn()
  call gen_cdfnor()
  call gen_cdfpoi()
  call gen_cdft()

  close(trace_unit)

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

  function fmt_real(v) result(buf)
    real(kind=rk), intent(in) :: v
    character(len=16) :: buf
    character(len=16) :: solver_trace_hex64
    buf = solver_trace_hex64(v)
  end function fmt_real

  function fmt_int(v) result(buf)
    integer, intent(in) :: v
    character(len=32) :: buf
    write(buf, '(i0)') v
    buf = trim(adjustl(buf))
  end function fmt_int

  subroutine trace_case_begin(id, kind, meta)
    character(len=*), intent(in) :: id, kind, meta
    write(trace_unit, '(a)') 'case,id=' // trim(id) // ',kind=' // trim(kind) // ',' // trim(meta)
    call solver_trace_enable(trace_unit)
  end subroutine trace_case_begin

  subroutine trace_case_end(id)
    character(len=*), intent(in) :: id
    call solver_trace_disable()
    write(trace_unit, '(a)') 'end,id=' // trim(id)
  end subroutine trace_case_end

  ! cdfbet: Beta(a, b), x in (0, 1)
  subroutine gen_cdfbet()
    real(kind=rk), parameter :: rows(3, 11) = reshape((/ &
      0.10_rk,  0.5_rk,  0.5_rk, &
      0.50_rk,  0.5_rk,  0.5_rk, &
      0.90_rk,  0.5_rk,  0.5_rk, &
      0.10_rk,  2.0_rk,  3.0_rk, &
      0.50_rk,  2.0_rk,  3.0_rk, &
      0.90_rk,  2.0_rk,  3.0_rk, &
      0.01_rk,  5.0_rk,  1.0_rk, &
      0.99_rk,  1.0_rk,  5.0_rk, &
      0.50_rk, 10.0_rk, 10.0_rk, &
      0.30_rk, 50.0_rk, 25.0_rk, &
      0.70_rk, 25.0_rk, 50.0_rk /), (/3, 11/))
    character(len=64) :: id
    integer :: i, fx, fa, fb, which, status
    character(len=256) :: meta
    real(kind=rk) :: x, a, b, y, p, q, p_sf, bound, xx, yy, aa, bb

    open(newunit=fx, file='tests/data/cdfbet_x.csv', status='replace', action='write')
    open(newunit=fa, file='tests/data/cdfbet_a.csv', status='replace', action='write')
    open(newunit=fb, file='tests/data/cdfbet_b.csv', status='replace', action='write')
    write(fx, '(a)') '# p, q, a, b, x'
    write(fa, '(a)') '# p, q, x, b, a'
    write(fb, '(a)') '# p, q, x, a, b'
    do i = 1, 11
      x = rows(1, i); a = rows(2, i); b = rows(3, i)
      y = 1.0_rk - x
      which = 1
      call cdfbet(which, p, q, x, y, a, b, status, bound)
      if (status /= 0) cycle
      ! which=2: solve x
      xx = 0.5_rk; yy = 0.5_rk; which = 2
      call cdfbet(which, p, q, xx, yy, a, b, status, bound)
      if (status == 0) then
        call putval(fx, p, .false.)
        call putval(fx, q, .false.)
        call putval(fx, a, .false.)
        call putval(fx, b, .false.)
        call putval(fx, xx, .true.)
        write(id, '(a,i0)') 'cdfbet_x_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',a=' // fmt_real(a) // ',b=' // fmt_real(b)
        call trace_case_begin(id, 'cdfbet_x', meta)
        xx = 0.5_rk; yy = 0.5_rk; which = 2
        call cdfbet(which, p, q, xx, yy, a, b, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfbet_x_sf_', i
        meta = 'q=' // fmt_real(q) // ',a=' // fmt_real(a) // ',b=' // fmt_real(b)
        call trace_case_begin(id, 'cdfbet_x_sf', meta)
        p_sf = 1.0_rk - q
        xx = 0.5_rk; yy = 0.5_rk; which = 2
        call cdfbet(which, p_sf, q, xx, yy, a, b, status, bound)
        call trace_case_end(id)
      end if
      ! which=3: solve a
      aa = 5.0_rk; which = 3
      call cdfbet(which, p, q, x, y, aa, b, status, bound)
      if (status == 0) then
        call putval(fa, p, .false.)
        call putval(fa, q, .false.)
        call putval(fa, x, .false.)
        call putval(fa, b, .false.)
        call putval(fa, aa, .true.)
        write(id, '(a,i0)') 'cdfbet_a_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',b=' // fmt_real(b)
        call trace_case_begin(id, 'cdfbet_a', meta)
        aa = 5.0_rk; which = 3
        call cdfbet(which, p, q, x, y, aa, b, status, bound)
        call trace_case_end(id)
      end if
      ! which=4: solve b
      bb = 5.0_rk; which = 4
      call cdfbet(which, p, q, x, y, a, bb, status, bound)
      if (status == 0) then
        call putval(fb, p, .false.)
        call putval(fb, q, .false.)
        call putval(fb, x, .false.)
        call putval(fb, a, .false.)
        call putval(fb, bb, .true.)
        write(id, '(a,i0)') 'cdfbet_b_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',a=' // fmt_real(a)
        call trace_case_begin(id, 'cdfbet_b', meta)
        bb = 5.0_rk; which = 4
        call cdfbet(which, p, q, x, y, a, bb, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fx); close(fa); close(fb)
  end subroutine gen_cdfbet

  subroutine gen_cdfbin()
    real(kind=rk), parameter :: rows(3, 8) = reshape((/ &
       3.0_rk,  10.0_rk, 0.3_rk, &
       5.0_rk,  10.0_rk, 0.5_rk, &
       8.0_rk,  10.0_rk, 0.7_rk, &
      10.0_rk,  50.0_rk, 0.2_rk, &
      25.0_rk,  50.0_rk, 0.5_rk, &
      40.0_rk,  50.0_rk, 0.8_rk, &
      20.0_rk, 100.0_rk, 0.25_rk, &
      50.0_rk, 100.0_rk, 0.5_rk /), (/3, 8/))
    character(len=64) :: id
    integer :: i, fs, fxn, fpr, which, status
    character(len=256) :: meta
    real(kind=rk) :: s, xn, pr, ompr, p, q, p_sf, bound, ss, xx_xn, pp, qq

    open(newunit=fs,  file='tests/data/cdfbin_s.csv',  status='replace', action='write')
    open(newunit=fxn, file='tests/data/cdfbin_xn.csv', status='replace', action='write')
    open(newunit=fpr, file='tests/data/cdfbin_pr.csv', status='replace', action='write')
    write(fs,  '(a)') '# p, q, xn, pr, s'
    write(fxn, '(a)') '# p, q, s, pr, xn'
    write(fpr, '(a)') '# p, q, s, xn, pr'
    do i = 1, 8
      s = rows(1, i); xn = rows(2, i); pr = rows(3, i)
      ompr = 1.0_rk - pr
      which = 1
      call cdfbin(which, p, q, s, xn, pr, ompr, status, bound)
      if (status /= 0) cycle
      ss = 0.0_rk; which = 2
      call cdfbin(which, p, q, ss, xn, pr, ompr, status, bound)
      if (status == 0) then
        call putval(fs, p, .false.); call putval(fs, q, .false.)
        call putval(fs, xn, .false.); call putval(fs, pr, .false.); call putval(fs, ss, .true.)
        write(id, '(a,i0)') 'cdfbin_s_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',xn=' // fmt_real(xn) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfbin_s', meta)
        ss = 0.0_rk; which = 2
        call cdfbin(which, p, q, ss, xn, pr, ompr, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfbin_s_sf_', i
        meta = 'q=' // fmt_real(q) // ',xn=' // fmt_real(xn) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfbin_s_sf', meta)
        p_sf = 1.0_rk - q
        ss = 0.0_rk; which = 2
        call cdfbin(which, p_sf, q, ss, xn, pr, ompr, status, bound)
        call trace_case_end(id)
      end if
      xx_xn = 5.0_rk; which = 3
      call cdfbin(which, p, q, s, xx_xn, pr, ompr, status, bound)
      if (status == 0) then
        call putval(fxn, p, .false.); call putval(fxn, q, .false.)
        call putval(fxn, s, .false.); call putval(fxn, pr, .false.); call putval(fxn, xx_xn, .true.)
        write(id, '(a,i0)') 'cdfbin_xn_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',s=' // fmt_real(s) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfbin_xn', meta)
        xx_xn = 5.0_rk; which = 3
        call cdfbin(which, p, q, s, xx_xn, pr, ompr, status, bound)
        call trace_case_end(id)
      end if
      pp = 0.5_rk; qq = 0.5_rk; which = 4
      call cdfbin(which, p, q, s, xn, pp, qq, status, bound)
      if (status == 0) then
        call putval(fpr, p, .false.); call putval(fpr, q, .false.)
        call putval(fpr, s, .false.); call putval(fpr, xn, .false.); call putval(fpr, pp, .true.)
        write(id, '(a,i0)') 'cdfbin_pr_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',s=' // fmt_real(s) // ',xn=' // fmt_real(xn)
        call trace_case_begin(id, 'cdfbin_pr', meta)
        pp = 0.5_rk; qq = 0.5_rk; which = 4
        call cdfbin(which, p, q, s, xn, pp, qq, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fs); close(fxn); close(fpr)
  end subroutine gen_cdfbin

  subroutine gen_cdfchi()
    real(kind=rk), parameter :: rows(2, 8) = reshape((/ &
        1.0_rk,   1.0_rk, &
        3.84_rk,  1.0_rk, &
        5.0_rk,   5.0_rk, &
       11.07_rk,  5.0_rk, &
       10.0_rk,  10.0_rk, &
       18.31_rk, 10.0_rk, &
      100.0_rk, 100.0_rk, &
      124.34_rk, 100.0_rk /), (/2, 8/))
    character(len=64) :: id
    integer :: i, fx, fdf, which, status
    character(len=256) :: meta
    real(kind=rk) :: x, df, p, q, p_sf, bound, xx, dd

    open(newunit=fx,  file='tests/data/cdfchi_x.csv',  status='replace', action='write')
    open(newunit=fdf, file='tests/data/cdfchi_df.csv', status='replace', action='write')
    write(fx,  '(a)') '# p, q, df, x'
    write(fdf, '(a)') '# p, q, x, df'
    do i = 1, 8
      x = rows(1, i); df = rows(2, i)
      which = 1
      call cdfchi(which, p, q, x, df, status, bound)
      if (status /= 0) cycle
      xx = 5.0_rk; which = 2
      call cdfchi(which, p, q, xx, df, status, bound)
      if (status == 0) then
        call putval(fx, p, .false.); call putval(fx, q, .false.)
        call putval(fx, df, .false.); call putval(fx, xx, .true.)
        write(id, '(a,i0)') 'cdfchi_x_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',df=' // fmt_real(df)
        call trace_case_begin(id, 'cdfchi_x', meta)
        xx = 5.0_rk; which = 2
        call cdfchi(which, p, q, xx, df, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfchi_x_sf_', i
        meta = 'q=' // fmt_real(q) // ',df=' // fmt_real(df)
        call trace_case_begin(id, 'cdfchi_x_sf', meta)
        p_sf = 1.0_rk - q
        xx = 5.0_rk; which = 2
        call cdfchi(which, p_sf, q, xx, df, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 3
      call cdfchi(which, p, q, x, dd, status, bound)
      if (status == 0) then
        call putval(fdf, p, .false.); call putval(fdf, q, .false.)
        call putval(fdf, x, .false.); call putval(fdf, dd, .true.)
        write(id, '(a,i0)') 'cdfchi_df_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x)
        call trace_case_begin(id, 'cdfchi_df', meta)
        dd = 5.0_rk; which = 3
        call cdfchi(which, p, q, x, dd, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fx); close(fdf)
  end subroutine gen_cdfchi

  subroutine gen_cdfchn()
    real(kind=rk), parameter :: rows(3, 8) = reshape((/ &
        5.0_rk,  5.0_rk,  2.0_rk, &
       10.0_rk,  5.0_rk,  2.0_rk, &
       20.0_rk,  5.0_rk,  2.0_rk, &
       15.0_rk,  5.0_rk, 10.0_rk, &
       20.0_rk, 10.0_rk,  5.0_rk, &
       30.0_rk, 10.0_rk,  5.0_rk, &
       50.0_rk, 20.0_rk,  5.0_rk, &
       25.0_rk,  5.0_rk, 20.0_rk /), (/3, 8/))
    character(len=64) :: id
    integer :: i, fx, fdf, fnc, which, status
    character(len=320) :: meta
    real(kind=rk) :: x, df, pnonc, p, q, bound, xx, dd, pp

    open(newunit=fx,  file='tests/data/cdfchn_x.csv',     status='replace', action='write')
    open(newunit=fdf, file='tests/data/cdfchn_df.csv',    status='replace', action='write')
    open(newunit=fnc, file='tests/data/cdfchn_pnonc.csv', status='replace', action='write')
    write(fx,  '(a)') '# p, q, df, pnonc, x'
    write(fdf, '(a)') '# p, q, x, pnonc, df'
    write(fnc, '(a)') '# p, q, x, df, pnonc'
    do i = 1, 8
      x = rows(1, i); df = rows(2, i); pnonc = rows(3, i)
      which = 1
      call cdfchn(which, p, q, x, df, pnonc, status, bound)
      if (status /= 0) cycle
      xx = 5.0_rk; which = 2
      call cdfchn(which, p, q, xx, df, pnonc, status, bound)
      if (status == 0) then
        call putval(fx, p, .false.); call putval(fx, q, .false.)
        call putval(fx, df, .false.); call putval(fx, pnonc, .false.); call putval(fx, xx, .true.)
        write(id, '(a,i0)') 'cdfchn_x_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',df=' // fmt_real(df) // ',pnonc=' // fmt_real(pnonc)
        call trace_case_begin(id, 'cdfchn_x', meta)
        xx = 5.0_rk; which = 2
        call cdfchn(which, p, q, xx, df, pnonc, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 3
      call cdfchn(which, p, q, x, dd, pnonc, status, bound)
      if (status == 0) then
        call putval(fdf, p, .false.); call putval(fdf, q, .false.)
        call putval(fdf, x, .false.); call putval(fdf, pnonc, .false.); call putval(fdf, dd, .true.)
        write(id, '(a,i0)') 'cdfchn_df_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',pnonc=' // fmt_real(pnonc)
        call trace_case_begin(id, 'cdfchn_df', meta)
        dd = 5.0_rk; which = 3
        call cdfchn(which, p, q, x, dd, pnonc, status, bound)
        call trace_case_end(id)
      end if
      pp = 5.0_rk; which = 4
      call cdfchn(which, p, q, x, df, pp, status, bound)
      if (status == 0) then
        call putval(fnc, p, .false.); call putval(fnc, q, .false.)
        call putval(fnc, x, .false.); call putval(fnc, df, .false.); call putval(fnc, pp, .true.)
        write(id, '(a,i0)') 'cdfchn_pnonc_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',df=' // fmt_real(df)
        call trace_case_begin(id, 'cdfchn_pnonc', meta)
        pp = 5.0_rk; which = 4
        call cdfchn(which, p, q, x, df, pp, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fx); close(fdf); close(fnc)
  end subroutine gen_cdfchn

  subroutine gen_cdff()
    real(kind=rk), parameter :: rows(3, 7) = reshape((/ &
      1.0_rk,   5.0_rk,  10.0_rk, &
      3.33_rk,  5.0_rk,  10.0_rk, &
      0.5_rk,   5.0_rk,  10.0_rk, &
      2.0_rk,  10.0_rk,  20.0_rk, &
      4.0_rk,   2.0_rk,  30.0_rk, &
      1.5_rk,  30.0_rk,  30.0_rk, &
      3.0_rk,  50.0_rk, 100.0_rk /), (/3, 7/))
    character(len=64) :: id
    integer :: i, ff, fdn, fdd, which, status
    character(len=320) :: meta
    real(kind=rk) :: f, dfn, dfd, p, q, p_sf, bound, ff_, dd

    open(newunit=ff,  file='tests/data/cdff_f.csv',   status='replace', action='write')
    open(newunit=fdn, file='tests/data/cdff_dfn.csv', status='replace', action='write')
    open(newunit=fdd, file='tests/data/cdff_dfd.csv', status='replace', action='write')
    write(ff,  '(a)') '# p, q, dfn, dfd, f'
    write(fdn, '(a)') '# p, q, f, dfd, dfn'
    write(fdd, '(a)') '# p, q, f, dfn, dfd'
    do i = 1, 7
      f = rows(1, i); dfn = rows(2, i); dfd = rows(3, i)
      which = 1
      call cdff(which, p, q, f, dfn, dfd, status, bound)
      if (status /= 0) cycle
      ff_ = 5.0_rk; which = 2
      call cdff(which, p, q, ff_, dfn, dfd, status, bound)
      if (status == 0) then
        call putval(ff, p, .false.); call putval(ff, q, .false.)
        call putval(ff, dfn, .false.); call putval(ff, dfd, .false.); call putval(ff, ff_, .true.)
        write(id, '(a,i0)') 'cdff_f_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',dfn=' // fmt_real(dfn) // ',dfd=' // fmt_real(dfd)
        call trace_case_begin(id, 'cdff_f', meta)
        ff_ = 5.0_rk; which = 2
        call cdff(which, p, q, ff_, dfn, dfd, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdff_f_sf_', i
        meta = 'q=' // fmt_real(q) // ',dfn=' // fmt_real(dfn) // ',dfd=' // fmt_real(dfd)
        call trace_case_begin(id, 'cdff_f_sf', meta)
        p_sf = 1.0_rk - q
        ff_ = 5.0_rk; which = 2
        call cdff(which, p_sf, q, ff_, dfn, dfd, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 3
      call cdff(which, p, q, f, dd, dfd, status, bound)
      if (status == 0) then
        call putval(fdn, p, .false.); call putval(fdn, q, .false.)
        call putval(fdn, f, .false.); call putval(fdn, dfd, .false.); call putval(fdn, dd, .true.)
        write(id, '(a,i0)') 'cdff_dfn_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',f=' // fmt_real(f) // ',dfd=' // fmt_real(dfd)
        call trace_case_begin(id, 'cdff_dfn', meta)
        dd = 5.0_rk; which = 3
        call cdff(which, p, q, f, dd, dfd, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 4
      call cdff(which, p, q, f, dfn, dd, status, bound)
      if (status == 0) then
        call putval(fdd, p, .false.); call putval(fdd, q, .false.)
        call putval(fdd, f, .false.); call putval(fdd, dfn, .false.); call putval(fdd, dd, .true.)
        write(id, '(a,i0)') 'cdff_dfd_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',f=' // fmt_real(f) // ',dfn=' // fmt_real(dfn)
        call trace_case_begin(id, 'cdff_dfd', meta)
        dd = 5.0_rk; which = 4
        call cdff(which, p, q, f, dfn, dd, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(ff); close(fdn); close(fdd)
  end subroutine gen_cdff

  subroutine gen_cdffnc()
    real(kind=rk), parameter :: rows(4, 4) = reshape((/ &
      1.0_rk,   5.0_rk,  10.0_rk, 2.0_rk, &
      4.0_rk,   5.0_rk,  10.0_rk, 2.0_rk, &
      2.0_rk,  10.0_rk,  20.0_rk, 5.0_rk, &
      3.0_rk,  20.0_rk,  30.0_rk, 10.0_rk /), (/4, 4/))
    character(len=64) :: id
    integer :: i, ff, fdn, fdd, fnc, which, status
    character(len=384) :: meta
    real(kind=rk) :: f, dfn, dfd, pnonc, p, q, bound, ff_, dd, pp

    open(newunit=ff,  file='tests/data/cdffnc_f.csv',     status='replace', action='write')
    open(newunit=fdn, file='tests/data/cdffnc_dfn.csv',   status='replace', action='write')
    open(newunit=fdd, file='tests/data/cdffnc_dfd.csv',   status='replace', action='write')
    open(newunit=fnc, file='tests/data/cdffnc_phonc.csv', status='replace', action='write')
    write(ff,  '(a)') '# p, q, dfn, dfd, phonc, f'
    write(fdn, '(a)') '# p, q, f, dfd, phonc, dfn'
    write(fdd, '(a)') '# p, q, f, dfn, phonc, dfd'
    write(fnc, '(a)') '# p, q, f, dfn, dfd, phonc'
    do i = 1, 4
      f = rows(1, i); dfn = rows(2, i); dfd = rows(3, i); pnonc = rows(4, i)
      which = 1
      call cdffnc(which, p, q, f, dfn, dfd, pnonc, status, bound)
      if (status /= 0) cycle
      ff_ = 5.0_rk; which = 2
      call cdffnc(which, p, q, ff_, dfn, dfd, pnonc, status, bound)
      if (status == 0) then
        call putval(ff, p, .false.); call putval(ff, q, .false.)
        call putval(ff, dfn, .false.); call putval(ff, dfd, .false.)
        call putval(ff, pnonc, .false.); call putval(ff, ff_, .true.)
        write(id, '(a,i0)') 'cdffnc_f_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',dfn=' // fmt_real(dfn) // ',dfd=' // fmt_real(dfd) // ',pnonc=' // fmt_real(pnonc)
        call trace_case_begin(id, 'cdffnc_f', meta)
        ff_ = 5.0_rk; which = 2
        call cdffnc(which, p, q, ff_, dfn, dfd, pnonc, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 3
      call cdffnc(which, p, q, f, dd, dfd, pnonc, status, bound)
      if (status == 0) then
        call putval(fdn, p, .false.); call putval(fdn, q, .false.)
        call putval(fdn, f, .false.); call putval(fdn, dfd, .false.)
        call putval(fdn, pnonc, .false.); call putval(fdn, dd, .true.)
        write(id, '(a,i0)') 'cdffnc_dfn_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',f=' // fmt_real(f) // ',dfd=' // fmt_real(dfd) // ',pnonc=' // fmt_real(pnonc)
        call trace_case_begin(id, 'cdffnc_dfn', meta)
        dd = 5.0_rk; which = 3
        call cdffnc(which, p, q, f, dd, dfd, pnonc, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 4
      call cdffnc(which, p, q, f, dfn, dd, pnonc, status, bound)
      if (status == 0) then
        call putval(fdd, p, .false.); call putval(fdd, q, .false.)
        call putval(fdd, f, .false.); call putval(fdd, dfn, .false.)
        call putval(fdd, pnonc, .false.); call putval(fdd, dd, .true.)
        write(id, '(a,i0)') 'cdffnc_dfd_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',f=' // fmt_real(f) // ',dfn=' // fmt_real(dfn) // ',pnonc=' // fmt_real(pnonc)
        call trace_case_begin(id, 'cdffnc_dfd', meta)
        dd = 5.0_rk; which = 4
        call cdffnc(which, p, q, f, dfn, dd, pnonc, status, bound)
        call trace_case_end(id)
      end if
      pp = 5.0_rk; which = 5
      call cdffnc(which, p, q, f, dfn, dfd, pp, status, bound)
      if (status == 0) then
        call putval(fnc, p, .false.); call putval(fnc, q, .false.)
        call putval(fnc, f, .false.); call putval(fnc, dfn, .false.)
        call putval(fnc, dfd, .false.); call putval(fnc, pp, .true.)
        write(id, '(a,i0)') 'cdffnc_phonc_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',f=' // fmt_real(f) // ',dfn=' // fmt_real(dfn) // ',dfd=' // fmt_real(dfd)
        call trace_case_begin(id, 'cdffnc_phonc', meta)
        pp = 5.0_rk; which = 5
        call cdffnc(which, p, q, f, dfn, dfd, pp, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(ff); close(fdn); close(fdd); close(fnc)
  end subroutine gen_cdffnc

  subroutine gen_cdfgam()
    real(kind=rk), parameter :: rows(3, 12) = reshape((/ &
       0.5_rk,  1.0_rk, 1.0_rk, &
       1.5_rk,  1.0_rk, 1.0_rk, &
       3.0_rk,  2.0_rk, 1.0_rk, &
       5.0_rk,  2.0_rk, 2.0_rk, &
       10.0_rk,  5.0_rk, 1.0_rk, &
       25.0_rk, 10.0_rk, 2.0_rk, &
       0.15_rk, 0.15_rk, 1.0_rk, &
       0.8_rk,  0.8_rk,  1.0_rk, &
       2.25_rk, 2.25_rk, 1.0_rk, &
       10.0_rk, 10.0_rk, 1.0_rk, &
       20.0_rk, 20.0_rk, 1.0_rk, &
       20.0_rk, 0.8_rk,  20.0_rk /), (/3, 12/))
    character(len=64) :: id
    integer :: i, fx, fsh, fsc, which, status
    character(len=320) :: meta
    real(kind=rk) :: x, shape, scale, p, q, p_sf, bound, xx, ss, sc

    open(newunit=fx,  file='tests/data/cdfgam_x.csv',     status='replace', action='write')
    open(newunit=fsh, file='tests/data/cdfgam_shape.csv', status='replace', action='write')
    open(newunit=fsc, file='tests/data/cdfgam_scale.csv', status='replace', action='write')
    write(fx,  '(a)') '# p, q, shape, scale, x'
    write(fsh, '(a)') '# p, q, x, scale, shape'
    write(fsc, '(a)') '# p, q, x, shape, scale'
    do i = 1, 12
      x = rows(1, i); shape = rows(2, i); scale = rows(3, i)
      which = 1
      call cdfgam(which, p, q, x, shape, scale, status, bound)
      if (status /= 0) cycle
      if (p <= 0.0_rk .or. p >= 1.0_rk .or. q <= 0.0_rk .or. q >= 1.0_rk) cycle
      xx = 5.0_rk; which = 2
      call cdfgam(which, p, q, xx, shape, scale, status, bound)
      if (status == 0) then
        call putval(fx, p, .false.); call putval(fx, q, .false.)
        call putval(fx, shape, .false.); call putval(fx, scale, .false.); call putval(fx, xx, .true.)
        write(id, '(a,i0)') 'cdfgam_x_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',shape=' // fmt_real(shape) // ',scale=' // fmt_real(scale)
        call trace_case_begin(id, 'cdfgam_x', meta)
        xx = 5.0_rk; which = 2
        call cdfgam(which, p, q, xx, shape, scale, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfgam_x_sf_', i
        meta = 'q=' // fmt_real(q) // ',shape=' // fmt_real(shape) // ',scale=' // fmt_real(scale)
        call trace_case_begin(id, 'cdfgam_x_sf', meta)
        p_sf = 1.0_rk - q
        xx = 5.0_rk; which = 2
        call cdfgam(which, p_sf, q, xx, shape, scale, status, bound)
        call trace_case_end(id)
      end if
      ss = 5.0_rk; which = 3
      call cdfgam(which, p, q, x, ss, scale, status, bound)
      if (status == 0) then
        call putval(fsh, p, .false.); call putval(fsh, q, .false.)
        call putval(fsh, x, .false.); call putval(fsh, scale, .false.); call putval(fsh, ss, .true.)
        write(id, '(a,i0)') 'cdfgam_shape_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',scale=' // fmt_real(scale)
        call trace_case_begin(id, 'cdfgam_shape', meta)
        ss = 5.0_rk; which = 3
        call cdfgam(which, p, q, x, ss, scale, status, bound)
        call trace_case_end(id)
      end if
      sc = 1.0_rk; which = 4
      call cdfgam(which, p, q, x, shape, sc, status, bound)
      if (status == 0) then
        call putval(fsc, p, .false.); call putval(fsc, q, .false.)
        call putval(fsc, x, .false.); call putval(fsc, shape, .false.); call putval(fsc, sc, .true.)
        write(id, '(a,i0)') 'cdfgam_scale_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',shape=' // fmt_real(shape)
        call trace_case_begin(id, 'cdfgam_scale', meta)
        sc = 1.0_rk; which = 4
        call cdfgam(which, p, q, x, shape, sc, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fx); close(fsh); close(fsc)
  end subroutine gen_cdfgam

  subroutine gen_cdfnbn()
    real(kind=rk), parameter :: rows(3, 5) = reshape((/ &
       5.0_rk, 10.0_rk, 0.5_rk, &
       2.0_rk,  5.0_rk, 0.3_rk, &
      10.0_rk, 20.0_rk, 0.4_rk, &
       0.0_rk,  5.0_rk, 0.5_rk, &
      20.0_rk, 50.0_rk, 0.6_rk /), (/3, 5/))
    character(len=64) :: id
    integer :: i, fs, fxn, fpr, which, status
    character(len=256) :: meta
    real(kind=rk) :: s, xn, pr, ompr, p, q, p_sf, bound, ss, xx, pp, qq

    open(newunit=fs,  file='tests/data/cdfnbn_s.csv',  status='replace', action='write')
    open(newunit=fxn, file='tests/data/cdfnbn_xn.csv', status='replace', action='write')
    open(newunit=fpr, file='tests/data/cdfnbn_pr.csv', status='replace', action='write')
    write(fs,  '(a)') '# p, q, xn, pr, s'
    write(fxn, '(a)') '# p, q, s, pr, xn'
    write(fpr, '(a)') '# p, q, s, xn, pr'
    do i = 1, 5
      s = rows(1, i); xn = rows(2, i); pr = rows(3, i)
      ompr = 1.0_rk - pr
      which = 1
      call cdfnbn(which, p, q, s, xn, pr, ompr, status, bound)
      if (status /= 0) cycle
      ss = 5.0_rk; which = 2
      call cdfnbn(which, p, q, ss, xn, pr, ompr, status, bound)
      if (status == 0) then
        call putval(fs, p, .false.); call putval(fs, q, .false.)
        call putval(fs, xn, .false.); call putval(fs, pr, .false.); call putval(fs, ss, .true.)
        write(id, '(a,i0)') 'cdfnbn_s_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',xn=' // fmt_real(xn) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfnbn_s', meta)
        ss = 5.0_rk; which = 2
        call cdfnbn(which, p, q, ss, xn, pr, ompr, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfnbn_s_sf_', i
        meta = 'q=' // fmt_real(q) // ',xn=' // fmt_real(xn) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfnbn_s_sf', meta)
        p_sf = 1.0_rk - q
        ss = 5.0_rk; which = 2
        call cdfnbn(which, p_sf, q, ss, xn, pr, ompr, status, bound)
        call trace_case_end(id)
      end if
      xx = 5.0_rk; which = 3
      call cdfnbn(which, p, q, s, xx, pr, ompr, status, bound)
      if (status == 0) then
        call putval(fxn, p, .false.); call putval(fxn, q, .false.)
        call putval(fxn, s, .false.); call putval(fxn, pr, .false.); call putval(fxn, xx, .true.)
        write(id, '(a,i0)') 'cdfnbn_xn_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',s=' // fmt_real(s) // ',pr=' // fmt_real(pr)
        call trace_case_begin(id, 'cdfnbn_xn', meta)
        xx = 5.0_rk; which = 3
        call cdfnbn(which, p, q, s, xx, pr, ompr, status, bound)
        call trace_case_end(id)
      end if
      pp = 0.5_rk; qq = 0.5_rk; which = 4
      call cdfnbn(which, p, q, s, xn, pp, qq, status, bound)
      if (status == 0) then
        call putval(fpr, p, .false.); call putval(fpr, q, .false.)
        call putval(fpr, s, .false.); call putval(fpr, xn, .false.); call putval(fpr, pp, .true.)
        write(id, '(a,i0)') 'cdfnbn_pr_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',s=' // fmt_real(s) // ',xn=' // fmt_real(xn)
        call trace_case_begin(id, 'cdfnbn_pr', meta)
        pp = 0.5_rk; qq = 0.5_rk; which = 4
        call cdfnbn(which, p, q, s, xn, pp, qq, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fs); close(fxn); close(fpr)
  end subroutine gen_cdfnbn

  subroutine gen_cdfnor()
    real(kind=rk), parameter :: rows(3, 5) = reshape((/ &
      -2.0_rk, 0.0_rk, 1.0_rk, &
       0.0_rk, 0.0_rk, 1.0_rk, &
       1.96_rk, 0.0_rk, 1.0_rk, &
       5.0_rk, 3.0_rk, 2.0_rk, &
      -1.0_rk, 1.0_rk, 0.5_rk /), (/3, 5/))
    character(len=64) :: id
    integer :: i, fx, fmean, fsd, which, status
    character(len=256) :: meta
    real(kind=rk) :: x, mean, sd, p, q, p_sf, bound, xx, mm, ss

    open(newunit=fx,    file='tests/data/cdfnor_x.csv',    status='replace', action='write')
    open(newunit=fmean, file='tests/data/cdfnor_mean.csv', status='replace', action='write')
    open(newunit=fsd,   file='tests/data/cdfnor_sd.csv',   status='replace', action='write')
    write(fx,    '(a)') '# p, q, mean, sd, x'
    write(fmean, '(a)') '# p, q, x, sd, mean'
    write(fsd,   '(a)') '# p, q, x, mean, sd'
    do i = 1, 5
      x = rows(1, i); mean = rows(2, i); sd = rows(3, i)
      which = 1
      call cdfnor(which, p, q, x, mean, sd, status, bound)
      if (status /= 0) cycle
      xx = 0.0_rk; which = 2
      call cdfnor(which, p, q, xx, mean, sd, status, bound)
      if (status == 0) then
        call putval(fx, p, .false.); call putval(fx, q, .false.)
        call putval(fx, mean, .false.); call putval(fx, sd, .false.); call putval(fx, xx, .true.)
        write(id, '(a,i0)') 'cdfnor_x_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',mean=' // fmt_real(mean) // ',sd=' // fmt_real(sd)
        call trace_case_begin(id, 'cdfnor_x', meta)
        xx = 0.0_rk; which = 2
        call cdfnor(which, p, q, xx, mean, sd, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfnor_x_sf_', i
        meta = 'q=' // fmt_real(q) // ',mean=' // fmt_real(mean) // ',sd=' // fmt_real(sd)
        call trace_case_begin(id, 'cdfnor_x_sf', meta)
        p_sf = 1.0_rk - q
        xx = 0.0_rk; which = 2
        call cdfnor(which, p_sf, q, xx, mean, sd, status, bound)
        call trace_case_end(id)
      end if
      mm = 0.0_rk; which = 3
      call cdfnor(which, p, q, x, mm, sd, status, bound)
      if (status == 0) then
        call putval(fmean, p, .false.); call putval(fmean, q, .false.)
        call putval(fmean, x, .false.); call putval(fmean, sd, .false.); call putval(fmean, mm, .true.)
        write(id, '(a,i0)') 'cdfnor_mean_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',sd=' // fmt_real(sd)
        call trace_case_begin(id, 'cdfnor_mean', meta)
        mm = 0.0_rk; which = 3
        call cdfnor(which, p, q, x, mm, sd, status, bound)
        call trace_case_end(id)
      end if
      ss = 1.0_rk; which = 4
      call cdfnor(which, p, q, x, mean, ss, status, bound)
      if (status == 0) then
        call putval(fsd, p, .false.); call putval(fsd, q, .false.)
        call putval(fsd, x, .false.); call putval(fsd, mean, .false.); call putval(fsd, ss, .true.)
        write(id, '(a,i0)') 'cdfnor_sd_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',x=' // fmt_real(x) // ',mean=' // fmt_real(mean)
        call trace_case_begin(id, 'cdfnor_sd', meta)
        ss = 1.0_rk; which = 4
        call cdfnor(which, p, q, x, mean, ss, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fx); close(fmean); close(fsd)
  end subroutine gen_cdfnor

  subroutine gen_cdfpoi()
    real(kind=rk), parameter :: rows(2, 7) = reshape((/ &
        0.0_rk,   1.0_rk, &
        2.0_rk,   3.0_rk, &
        5.0_rk,   3.0_rk, &
       10.0_rk,   5.0_rk, &
       50.0_rk,  60.0_rk, &
      100.0_rk, 100.0_rk, &
        3.0_rk,   7.7537_rk /), (/2, 7/))
    character(len=64) :: id
    integer :: i, fs, fl, which, status
    character(len=256) :: meta
    real(kind=rk) :: s, xlam, p, q, p_sf, bound, ss, ll

    open(newunit=fs, file='tests/data/cdfpoi_s.csv',    status='replace', action='write')
    open(newunit=fl, file='tests/data/cdfpoi_xlam.csv', status='replace', action='write')
    write(fs, '(a)') '# p, q, xlam, s'
    write(fl, '(a)') '# p, q, s, xlam'
    do i = 1, 7
      s = rows(1, i); xlam = rows(2, i)
      which = 1
      call cdfpoi(which, p, q, s, xlam, status, bound)
      if (status /= 0) cycle
      ss = 5.0_rk; which = 2
      call cdfpoi(which, p, q, ss, xlam, status, bound)
      if (status == 0) then
        call putval(fs, p, .false.); call putval(fs, q, .false.)
        call putval(fs, xlam, .false.); call putval(fs, ss, .true.)
        write(id, '(a,i0)') 'cdfpoi_s_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',xlam=' // fmt_real(xlam)
        call trace_case_begin(id, 'cdfpoi_s', meta)
        ss = 5.0_rk; which = 2
        call cdfpoi(which, p, q, ss, xlam, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdfpoi_s_sf_', i
        meta = 'q=' // fmt_real(q) // ',xlam=' // fmt_real(xlam)
        call trace_case_begin(id, 'cdfpoi_s_sf', meta)
        p_sf = 1.0_rk - q
        ss = 5.0_rk; which = 2
        call cdfpoi(which, p_sf, q, ss, xlam, status, bound)
        call trace_case_end(id)
      end if
      ll = 5.0_rk; which = 3
      call cdfpoi(which, p, q, s, ll, status, bound)
      if (status == 0) then
        call putval(fl, p, .false.); call putval(fl, q, .false.)
        call putval(fl, s, .false.); call putval(fl, ll, .true.)
        write(id, '(a,i0)') 'cdfpoi_xlam_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',s=' // fmt_real(s)
        call trace_case_begin(id, 'cdfpoi_xlam', meta)
        ll = 5.0_rk; which = 3
        call cdfpoi(which, p, q, s, ll, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(fs); close(fl)
  end subroutine gen_cdfpoi

  subroutine gen_cdft()
    real(kind=rk), parameter :: rows(2, 11) = reshape((/ &
        0.0_rk,   1.0_rk, &
        1.0_rk,   1.0_rk, &
       -1.0_rk,   1.0_rk, &
        2.0_rk,   5.0_rk, &
        2.776_rk, 4.0_rk, &
        -2.0_rk,  10.0_rk, &
        3.0_rk, 100.0_rk, &
        -1.96_rk, 1000.0_rk, &
         0.5_rk,   1.000000000001_rk, &
         0.5_rk,   2.0_rk, &
         0.5_rk,   2.000000000001_rk /), (/2, 11/))
    character(len=64) :: id
    integer :: i, ft, fdf, which, status
    character(len=256) :: meta
    real(kind=rk) :: t, df, p, q, p_sf, bound, tt, dd

    open(newunit=ft,  file='tests/data/cdft_t.csv',  status='replace', action='write')
    open(newunit=fdf, file='tests/data/cdft_df.csv', status='replace', action='write')
    write(ft,  '(a)') '# p, q, df, t'
    write(fdf, '(a)') '# p, q, t, df'
    do i = 1, 11
      t = rows(1, i); df = rows(2, i)
      which = 1
      call cdft(which, p, q, t, df, status, bound)
      if (status /= 0) cycle
      tt = 0.0_rk; which = 2
      call cdft(which, p, q, tt, df, status, bound)
      if (status == 0) then
        call putval(ft, p, .false.); call putval(ft, q, .false.)
        call putval(ft, df, .false.); call putval(ft, tt, .true.)
        write(id, '(a,i0)') 'cdft_t_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',df=' // fmt_real(df)
        call trace_case_begin(id, 'cdft_t', meta)
        tt = 0.0_rk; which = 2
        call cdft(which, p, q, tt, df, status, bound)
        call trace_case_end(id)
        write(id, '(a,i0)') 'cdft_t_sf_', i
        meta = 'q=' // fmt_real(q) // ',df=' // fmt_real(df)
        call trace_case_begin(id, 'cdft_t_sf', meta)
        p_sf = 1.0_rk - q
        tt = 0.0_rk; which = 2
        call cdft(which, p_sf, q, tt, df, status, bound)
        call trace_case_end(id)
      end if
      dd = 5.0_rk; which = 3
      call cdft(which, p, q, t, dd, status, bound)
      if (status == 0) then
        call putval(fdf, p, .false.); call putval(fdf, q, .false.)
        call putval(fdf, t, .false.); call putval(fdf, dd, .true.)
        write(id, '(a,i0)') 'cdft_df_', i
        meta = 'p=' // fmt_real(p) // ',q=' // fmt_real(q) // ',t=' // fmt_real(t)
        call trace_case_begin(id, 'cdft_df', meta)
        dd = 5.0_rk; which = 3
        call cdft(which, p, q, t, dd, status, bound)
        call trace_case_end(id)
      end if
    end do
    close(ft); close(fdf)
  end subroutine gen_cdft

end program gen_dispatchers
