# discover_schema.py
import ijson
from collections import defaultdict
import yaml

def discover_schema(file_path):
    schema = defaultdict(set)

    with open(file_path, 'rb') as f:
        parser = ijson.parse(f)
        current_path = []
        for prefix, event, value in parser:
            current_path = prefix.split('.')
            full_path = '.'.join(current_path)

            if event == 'map_key':
                current_path.append(value)
                continue
            elif event in ('start_map', 'start_array'):
                pass
            elif event in ('end_map', 'end_array'):
                if current_path:
                    current_path.pop()
                continue

            # leaf value
            if value is None:
                type_name = 'null'
            elif event == 'string':
                type_name = 'string'
            elif event in ('number', 'integer'):
                type_name = 'number'
            elif event == 'boolean':
                type_name = 'boolean'
            else:
                type_name = event

            schema['.'.join(current_path)].add(type_name)

    # Convert to nice nested dict
    nested = {}
    for path, types in schema.items():
        parts = path.split('.')
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {'_type': 'object', '_children': {}}
            elif '_children' not in d[part]:
                d[part]['_children'] = {}
            d = d[part]['_children']
        key = parts[-1]
        d[key] = {'_type': list(types)} if len(types) > 1 else {'_type': list(types)[0]}

    return nested

if __name__ == '__main__':
    import sys
    schema = discover_schema(sys.argv[1])
    print(yaml.dump(schema, default_flow_style=False, sort_keys=False))