"""Pinned bytes with explicit deferred annotations; not full sQUlearn runtime."""
import ast,hashlib,importlib,importlib.util,inspect,json,sys,types,__future__
from pathlib import Path
import numpy as np
from qiskit.circuit import ParameterVector
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'vendor/src/squlearn'
DEFERRED_MODULE='squlearn.encoding_circuit.layered_encoding_circuit'
DEFERRED_PATH='vendor/src/squlearn/encoding_circuit/layered_encoding_circuit.py'
RANDOM_LAYERED_MODULE='squlearn.encoding_circuit.circuit_library.random_layered_encoding_circuit'
DEFERRED_MODULES={
    DEFERRED_MODULE:DEFERRED_PATH,
    RANDOM_LAYERED_MODULE:'vendor/src/squlearn/encoding_circuit/circuit_library/random_layered_encoding_circuit.py'}
REQUIRES={
 'chebyshev_tower':'Mandatory num_chebyshev is unspecified by the fixed default-construction protocol.',
 'qiskit_encoding_circuit':'Wrapper requires a supplied qiskit_circuit; none is selected by this protocol.',
 'qcnn_encoding_circuit':'Default constructor is an empty operation builder; no custom/default=True architecture is selected.'}

def verify_sources():
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for e in manifest['files']:
        assert hashlib.sha256((ROOT/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
    return manifest

def bootstrap():
    verify_sources()
    for name,relative in [('squlearn',''),('squlearn.util','util'),('squlearn.encoding_circuit','encoding_circuit'),('squlearn.encoding_circuit.circuit_library','encoding_circuit/circuit_library')]:
        if name in sys.modules:
            assert getattr(sys.modules[name],'_coverage_host',False),'Run in a fresh Python process'
            continue
        module=types.ModuleType(name);module.__path__=[str(SOURCE/relative)];module._coverage_host=True
        sys.modules[name]=module

def compatibility_metadata():
    manifest=verify_sources()
    hashes={e['path']:e['sha256'] for e in manifest['files']}
    return {'modules':[{'module':name,'path':path,'sha256':hashes[path]} for name,path in DEFERRED_MODULES.items()],
            'compiler_flag':__future__.annotations.compiler_flag,
            'mode':'deferred_annotations_only','vendor_bytes_modified':False}

def load_deferred_dependency(name=DEFERRED_MODULE):
    """Compile exact verified bytes; do not use or generate bytecode caches."""
    if name not in DEFERRED_MODULES:
        raise ValueError('Only explicitly pinned compatibility modules are allowed')
    metadata=next(e for e in compatibility_metadata()['modules'] if e['module']==name)
    if name in sys.modules:
        module=sys.modules[name]
        if not getattr(module,'_coverage_deferred_annotations',False):
            raise RuntimeError('Dependency already loaded without compatibility policy; use a fresh process')
        return module
    if name==RANDOM_LAYERED_MODULE:
        load_deferred_dependency(DEFERRED_MODULE)
    path=ROOT/metadata['path']
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=metadata['sha256']:
        raise RuntimeError('Pinned dependency hash mismatch')
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec)
    code=compile(raw,str(path),'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True)
    sys.modules[name]=module
    try:
        exec(code,module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    module._coverage_deferred_annotations=True
    parent,child=name.rsplit('.',1)
    setattr(sys.modules[parent],child,module)
    return module

def inventory():
    out=[]
    for p in sorted((SOURCE/'encoding_circuit/circuit_library').glob('*.py')):
        if p.stem=='__init__':continue
        tree=ast.parse(p.read_text());classes=[x for x in tree.body if isinstance(x,ast.ClassDef)]
        assert len(classes)==1,p
        cls=classes[0];init=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=='__init__')
        out.append({'module':p.stem,'class':cls.name,'path':str(p.relative_to(ROOT)),
                    'class_line':cls.lineno,'constructor_line':init.lineno,
                    'has_layers':'num_layers' in [a.arg for a in init.args.args],
                    'configuration_required':REQUIRES.get(p.stem)})
    assert len(out)==13
    return out

def construct(entry,n,layers):
    bootstrap()
    if entry['module'] in ['kyriienko_nonlinear_encoding_circuit','random_layered_encoding_circuit']:
        load_deferred_dependency()
    if entry['module']=='random_layered_encoding_circuit':
        load_deferred_dependency(RANDOM_LAYERED_MODULE)
    module=importlib.import_module('squlearn.encoding_circuit.circuit_library.'+entry['module'])
    cls=getattr(module,entry['class']);kwargs={'num_qubits':n}
    if entry['has_layers']:kwargs['num_layers']=layers
    obj=cls(**kwargs)
    initial=obj.generate_initial_parameters(num_features=2,seed=0);count=len(initial)
    x,t=ParameterVector('a',2),ParameterVector('t',count)
    circuit=obj.get_circuit(x,t)
    assert circuit.num_qubits==n
    return obj,circuit,x,t,kwargs
