P h o n o n - r e n o r m a l i z a t i o n   s c h e m e
                                                (applied to beryllium monolayer)

This example shows in detail how to perform a constrained density-functional
perturbation theory (cDFPT) [Nomura, Arita: PRB 92, 245108 (2015)] calculation
and renormalize the resulting phonons to reproduce the full DFPT.

We consider a triangular lattice of beryllium atoms, which is the most simple
(hypothetical) two-dimensional system found to be stable in DFPT. Additionally,
the paramters chosen for the ab-initio calculations are far from convergence.
This enables us to run everything on a computer with only two processors.


1   c D F P T   m o d i f i c a t i o n

We use Quantum ESPRESSO for all ab-inito calculations involved. Since cDFPT is
not implemented, the following modifications of the code have to be made:

First, we define a new variable 'cdfpt' which takes the index (1-based) of the
electronic band which represents the target subspace of cDFPT. Any value out of
the range of valid band indices triggers a normal DFPT calculation. A general
definition of the target subspace in terms of momenta and band indices (k, n)
and (k + q, m) is straightforward but disregarded here for simplicity.

A convenient place for the new variable is the module 'control_lr':
 ______________________
| LR_Modules/lrcom.f90 |_______________________________________________________
|                                                                              |
| 48    LOGICAL  :: lgamma         ! if .TRUE. this is a q=0 computation       |
| 49    LOGICAL  :: lrpa           ! if .TRUE. uses the Random Phace Approx... |
|  +    INTEGER  :: cdfpt          ! index of cDFPT target band                |
| 50    !                                                                      |
| 51  END MODULE control_lr                                                    |
|______________________________________________________________________________|

The actual cDFPT modification is very simple:
 ______________________________
| LR_Modules/orthogonalize.f90 |_______________________________________________
|                                                                              |
| 41    USE control_flags,    ONLY : gamma_only                                |
| 42    USE gvect,            ONLY : gstart                                    |
|  *    USE control_lr,       ONLY : alpha_pv, nbnd_occ, cdfpt                 |
| 44    USE dfpt_tetra_mod,   ONLY : dfpt_tetra_beta                           |
| ..                                                                           |
| 90             wg1 = wgauss ((ef-et(ibnd,ikk)) / degauss, ngauss)            |
| 91             w0g = w0gauss((ef-et(ibnd,ikk)) / degauss, ngauss) / degauss  |
| 92             DO jbnd = 1, nbnd                                             |
|  +                IF (cdfpt == ibnd .and. cdfpt == jbnd) THEN                |
|  +                   ps(jbnd, ibnd) = wg1 * ps(jbnd, ibnd)                   |
|  +                   CYCLE                                                   |
|  +                END IF                                                     |
| 93                wgp = wgauss ( (ef - et (jbnd, ikq) ) / degauss, ngauss)   |
| 94                deltae = et (jbnd, ikq) - et (ibnd, ikk)                   |
| 94                theta = wgauss (deltae / degauss, 0)                       |
|______________________________________________________________________________|

Finally, the only thing left to do is to make 'cdfpt' an input parameter which
is read on the I/O processor and then broadcast to the other processors.
 __________________________
| PHonon/PH/phq_readin.f90 |___________________________________________________
|                                                                              |
|  73    USE qpoint,        ONLY : nksq, xq                                    |
|   *    USE control_lr,    ONLY : lgamma, lrpa, cdfpt                         |
|  75                                                                          |
|  ..                                                                          |
| 110    !                                                                     |
|   *    NAMELIST / INPUTPH / cdfpt, tr2_ph, amass, alpha_mix, niter_ph, nm... |
| 112                         nat_todo, verbosity, iverbosity, outdir, epsi... |
|  ..                                                                          |
| 220    ! ... set default values for variables in namelist                    |
| 221    !                                                                     |
|   +    cdfpt        = 0                                                      |
| 222    tr2_ph       = 1.D-12                                                 |
|  ..                                                                          |
| 333    ! ...  broadcast all input variables                                  |
| 334    !                                                                     |
| 335    tmp_dir = trimcheck (outdir)                                          |
| 336    CALL bcast_ph_input ( )                                               |
|   +    CALL mp_bcast(cdfpt, meta_ionode_id, world_comm)                      |
| 337    CALL mp_bcast(nogg, meta_ionode_id, world_comm  )                     |
|______________________________________________________________________________|
