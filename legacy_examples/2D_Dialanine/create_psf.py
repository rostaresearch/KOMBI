from simtk.openmm import app
import openmm.app as omm_app
# Load PDB file
pdb = app.PDBFile('dialaA.pdb')

# Load CHARMM force field
forcefield = omm_app.ForceField('amber14-all.xml')#, 'amber14/tip3p.xml')

# Create system
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

# Save the psf file.
