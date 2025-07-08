! to solve eqn 5.13 in Murray thesis
!
! tan(ax) = b*cot(cx)
! tan(ax)*tan(cx) - b = 0
! f(x) = 0

! bisector method
! - find interval such that f(a) * f(b) < 0
! - compute midpoint, m
! - eval f(m)
! - if f(a) * f(m) < 0, root is between a and m
! - else it is between m and b
! - repeat until residual is small
! - return that m as the root

double precision function f(x, a, b, c)
    double precision, intent(in) :: a, b, c
    double precision, intent(in) :: x

! operates on a scalar value of x and returns one real*8
    f = dtan(x*a) * dtan(x*c) - b
end function f


subroutine interval(x, N, a, b, c, root)
!
    integer, intent(in) :: N
    double precision, intent(in) :: x(N)
    double precision, intent(in) :: a, b, c
    double precision, intent(out) :: root

! apparently you need to redeclare type of func in subroutine
    double precision f  ! without ::

! only calls bisect if the interval covers a root (sign change)
    do i = 1, N - 1
        if (f(x(i), a, b, c) * f(x(i+1), a, b, c) .lt. 0.0d0) then
            call bisect(f, x(i), x(i+1), a, b, c, root)
        end if
    end do

end subroutine interval


subroutine bisect(f, x1, x2, a, b, c, root)
!
    double precision, intent(in) :: x1, x2, a, b, c
    double precision, intent(out) :: root  ! the answer
! values of function evaluated at x1, etc, and midpoint
    double precision :: f1, f2, fm, m
    integer :: i
    
    double precision, parameter :: residual = 1.0d-8 
    integer, parameter :: max_iter = 10000
    
    double precision f

    f1 = f(x1, a, b, c)
    f2 = f(x2, a, b, c)

    if (f1 * f2 .gt. 0.0d0) then
        print *, "no root in this interval"
        root = x1  ! why
        return
    end if

    do i = 1, max_iter
        m = (x1 + x2) / 2.0d0
        fm = f(m, a, b, c)
    ! function is close enough to zero m can be approx the root
    ! not sure about the .or. keep testing
        if (abs(fm) .lt. residual .or. abs(x2 - x1) .lt. residual) then
            root = m
            return
        end if
        if (f1 * fm .lt. 0.0d0) then
            root = m
            f2 = fm
        else
            root = m
            f1 = fm
        end if
    end do

    root = m

end subroutine bisect


program main
    implicit none

    double precision, parameter :: a = 1
    double precision, parameter :: b = 1
    double precision, parameter :: c = 1

    ! defines the steps and range to search over
    integer, parameter :: N = 100000
    integer, parameter :: iter = 50
    double precision, parameter :: x0 = -2.5d0, xN = 2.5d0
    

    double precision :: x(N)
    double precision :: root(iter)
    double precision :: root_avg
    double precision :: start, end
    integer :: i

    ! initialise the array x to search over
    do i = 1, N
        x(i) = x0 + (xN - x0) * (i - 1) / (N - 1)
    end do

    call cpu_time(start)

    root_avg = 0.0
    do i = 1, iter
        call interval(x, N, a, b, c, root(i))
        root_avg = root_avg + root(i)
    end do
! this still only returns the last found root

    root_avg = root_avg / iter

    call cpu_time(end)

    print *, "=========================================="
    print *, "=============== root finder =============="
    print *, "=========================================="
    print *, "root =", root_avg
    print *, "=========================================="
    print *, "time avg =", (end-start)/iter

end program main
