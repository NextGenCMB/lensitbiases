!To transform a two-dimensional real array, out of place, you might use the following:
!
!        double precision in
!        dimension in(M,N)
!        double complex out
!        dimension out(M/2 + 1, N)
!        integer*8 plan
!
!        call dfftw_plan_dft_r2c_2d(plan,M,N,in,out,FFTW_ESTIMATE)
!        call dfftw_execute_dft_r2c(plan, in, out)
!        call dfftw_destroy_plan(plan)
! FIXME: f2py does not support user-defined types !?
module class_box
  implicit none
  private
  public :: Box, box_print

  double precision :: pi = 3.1415926535897931d0 ! Class-wide private constant

  type Box  ! squared box, with side length lside and npix pixels on a side
     double precision :: lside
     integer :: npix
  end type Box
contains

  subroutine box_print(this)
    type(Box), intent(in) :: this
    print *, 'Box: lside = ', this%lside, ' npix on a side = ', this%npix
  end subroutine box_print
end module

!program circle_test
!  use class_Circle
!  implicit none

!  type(Circle) :: c     ! Declare a variable of type Circle.
!  c = Circle(1.5)       ! Use the implicit constructor, radius = 1.5.
!  call circle_print(c)  ! Call a class subroutine
!end program circle_test