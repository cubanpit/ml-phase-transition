! TO COMPILE FORTRAN
! gfortran -O3 ising.f90 random.o

module configuration
save
integer(4) Nx, Ny, nh, which_lattice, nn, Tn
integer(4), allocatable :: ivic(:, :), ivict(:, :), ixpiy(:) ! nearest neighbors
integer(4), allocatable :: spin(:) ! spin (-1 or 1 ) or (boson 0 or 1 respectively) configuration
real(8) temperature, beta, Jh, Tstep, Tstart, Tc, E, magt
end module configuration

program ising
use configuration
implicit none
integer(4) i, j, k, t, N, iseedd, nther, nblks, msteps, blkOutput
real(8) blk_magt, blk_E, drand1, usTime

! if = 1 enable extra output in stderr
blkOutput = 1

! lattice dimensions
Nx = 30
Ny = 30
nh = Nx * Ny

! lattice type: 1 = square , 2 = triangular
which_lattice = 2
Jh = 1    ! coupling constant

! temperature
Tc = 4.0d0 / log(3.0d0)
Tstart = 2.0d0
Tn = 40
Tstep = 2.0d0 * (Tc - Tstart) / (Tn - 1.0d0)

! NOTE: every MC step cycle on every Nx*Ny lattice site
nther = 50   ! thermalization steps
nblks = 50      ! number of MC blocks to take measurements
msteps = 200  ! effective MC steps

if(which_lattice == 1) then
  nn = 4
  allocate(ivic(nh, nn)) !square
  allocate(ixpiy(nh))
  call square()
elseif(which_lattice == 2) then
  nn = 6
  allocate(ivic(nh, nn), ivict(nh, 4)) !triangular
  call triangular()
end if

! use cpu time in microseconds (if possible) as seed
call cpu_time(usTime)
iseedd = int(usTime*1000000)

!initializations
call rand_init(iseedd)
call initial()

!# thermalization
temperature = Tstart
do t = 1, Tn
  do i = 1, nther
    beta = 1.0d0 / temperature
    call runbin(msteps)
  end do

  !# run
  beta = 1.0d0 / temperature
  blk_magt = 0.0d0
  blk_E = 0.0d0
  do i = 1, nblks
    call runbin(msteps)
    call accmeasure(i)
    blk_magt = blk_magt + magt
    blk_E = blk_E + E / nh
  end do

  ! compute average value
  blk_magt = blk_magt / nblks
  blk_E = blk_E / nblks
  if (blkOutput == 1) then
    write(0, *) temperature, blk_E, blk_magt
  end if
  write(6, *) temperature
  do i = 1, nh
    write(6, '(I2,X)', advance='no') spin(i)
  end do
  write(6, *) ""

  temperature = temperature + Tstep
  call cpu_time(usTime)
  iseedd = int(usTime*1000000)
end do

end program ising


subroutine square()
use configuration
implicit none
integer(4) i, j, k, ii
ii = 1

do j = 1, Ny
  do i = 1, Nx
    !#right  (1)
    if (i == Nx) then
      ivic(ii, 1) = ii - (Nx - 1)
    else
      ivic(ii, 1) = ii + 1
    end if
    ! # up (2)
    if (j == Ny) then
      ivic(ii, 2) = ii - Ny * (Nx - 1)
    else
      ivic(ii, 2) = ii + Nx
    end if
    !#left (3)
    if (i == 1) then
      ivic(ii, 3) = ii + (Nx - 1)
    else
      ivic(ii, 3) = ii - 1
    end if
    !#down (4)
    if (j == 1) then
      ivic(ii, 4) = ii + (Ny -1 ) * Nx
    else
      ivic(ii, 4) = ii - Nx
    end if
    ixpiy(ii) = i + j
    ii = ii + 1
  end do
end do

end subroutine square


subroutine triangular()
use configuration
implicit none
integer(4)i,j,k,ii
ii=1

do j = 1, Ny
  do i = 1, Nx
    !#right  (1)
    if (i == Nx) then
       ivict(ii, 1) = ii - (Nx - 1)
    else
       ivict(ii, 1) = ii + 1
    end if
    ! # up (2)
    if (j == Ny) then
       ivict(ii, 2) = ii - Ny * (Nx - 1)
    else
       ivict(ii, 2) = ii + Nx
    end if
    !#left (3)
    if (i == 1) then
       ivict(ii, 3) = ii + (Nx - 1)
    else
       ivict(ii, 3) = ii - 1
    end if
    !#down (4)
    if (j == 1) then
       ivict(ii, 4) = ii + (Ny - 1) * Nx
    else
       ivict(ii, 4) = ii - Nx
    end if
    ii = ii + 1
  end do
end do

ivic = 0

ii = 1
do j = 1, Ny
  do i = 1, Nx
    ivic(ii, 1) = ivict(ii, 1)
    ivic(ii, 3) = ivict(ii, 2)
    ivic(ii, 4) = ivict(ii, 3)
    ivic(ii, 6) = ivict(ii, 4)
    ivic(ii, 2) = ivict(ivict(ii, 1), 2)
    ivic(ii, 5) = ivict(ivict(ii, 3), 4)
    ii = ii + 1
  end do
end do

end subroutine triangular


!------------------------
!initial configuration
!-----------------------
subroutine initial()

use configuration
implicit none
integer :: i, j
real(8) :: drand1, bu

allocate(spin(nh))

do i = 1, nh
  ! random initialization - hot start
  spin(i) = 2 * int(2. * drand1()) - 1
  ! constant initialization - cold start
  !spin(i) = 1
end do

!#initial energy
E = 0.0
do i = 1, nh
  bu = 0.0
  do j = 1, nn/2
    bu = bu + spin(ivic(i, j))
  end do
  E = E - bu * Jh * spin(i)
end do

end subroutine initial


subroutine runbin(msteps)
use configuration
implicit none
integer(4) i, j, k, kk, msteps
real(8) DE, r, drand1, mag

mag = 0.0
do i = 1, msteps
  do j = 1, nh
    k = int(drand1() * nh + 1)
    DE = 0
    do kk = 1, nn
      DE = DE + spin(ivic(k, kk))
    end do
    DE = 2 * DE * Jh * spin(k)

    !#metropolis
    if (DE < 0) then
      spin(k) = - spin(k)
      E = E + DE
    else
      r = drand1()
      if (r < exp(- beta * DE)) then
        spin(k) = - spin(k)
        E = E + DE
      end if
    end if
  end do

  if(which_lattice == 1 .and. Jh > 0.0) then
    mag = mag + abs(sum(dble(spin))) / nh
  else if(which_lattice == 1 .and. Jh < 0.0) then
    magt = 0.0d0
    do k = 1, nh
      magt = magt + dble(spin(k)) * (-1.0d0)**ixpiy(k)
    end do
    mag = mag + abs(magt) / nh
  else if(which_lattice == 2) then
    mag = mag + abs(sum(dble(spin))) / nh
  end if
end do

end subroutine runbin


subroutine accmeasure(i)
use configuration
implicit none
integer(4)i,j,k

if(which_lattice == 1 .and. Jh > 0.0) then
  magt = abs(sum(dble(spin))) / (Nx * Ny)
else if(which_lattice == 1 .and. Jh < 0.0) then
  magt = 0.0d0
  do k = 1, nh
    magt = magt + dble(spin(k)) * (-1.0d0)**ixpiy(k)
  end do
  magt = abs(magt) / (Nx * Ny)
else if(which_lattice == 2) then
  magt = abs(sum(dble(spin))) / (Nx * Ny)
end if

end subroutine accmeasure
